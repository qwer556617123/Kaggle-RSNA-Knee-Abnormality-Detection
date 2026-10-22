"""Bounded GPU compute probe; never produces a submission or claimed OOF.

Uses public uint8 corpus volumes, not uploaded private labels. Replicated 2D
slices deliberately avoid inventing unknown series adjacency in the legacy
44-slice cache. The number and shape of backbone forwards match the planned
2.5D recipe; this probe qualifies compute only, not DICOM or clinical validity.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .research_coatnet import build_clean_model, freeze_batchnorm_statistics, soft_weighted_bce, PRETRAIN_SHA
from .research_validation import runtime_upper_bound


def one_file(root, name):
    paths = list(Path(root).rglob(name))
    if len(paths) != 1:
        raise ValueError(f'expected exactly one {name}; found {len(paths)}')
    return paths[0]


def profiling_windows(volume, mask, count):
    if volume.ndim != 3 or mask.shape != (volume.shape[0],) or volume.dtype != np.uint8:
        raise ValueError('invalid public corpus shape/dtype')
    valid = np.flatnonzero(mask > 0)
    if not len(valid):
        raise ValueError('empty study; no silent timing fallback')
    centers = valid[np.linspace(0, len(valid)-1, count).round().astype(int)]
    x = torch.from_numpy(np.array(volume[centers], dtype=np.float32, copy=True))[:,None].repeat(1,3,1,1) / 255
    x = F.interpolate(x, (384,384), mode='bicubic', align_corners=False).clamp(0,1)
    return (x - .5) / .5


def run_profile(input_root='/kaggle/input', output_root='/kaggle/working/research_profile'):
    import timm
    started = time.perf_counter()
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'role': 'compute_profile_only', 'clinical_validation': False,
               'formal_submission_ready': False, 'decode_failures': None,
               'decode_branch': 'public_predecoded_corpus_no_dicom',
               'sampling': '96 studies stratified by valid-slice count; not language/protocol representative',
               'input_branch': 'replicated_2d_for_shape_equivalent_cost_only',
               'pretrain_sha256': PRETRAIN_SHA, 'train_windows': 12, 'eval_windows': 24,
               'resolution': 384, 'batch_size': 1, 'accumulation_steps': 8,
               'epochs_scenario': 3, 'torch': torch.__version__, 'timm': timm.__version__}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('GPU is required for meaningful compute profiling')
        torch.set_num_threads(2)
        torch.manual_seed(20260905)
        torch.cuda.manual_seed_all(20260905)
        torch.cuda.reset_peak_memory_stats()
        pretrain = one_file(input_root, 'pretrain_manifest.json').parent
        model = build_clean_model(pretrain).cuda()
        receipt['model_load_seconds'] = time.perf_counter() - started
        parts = []
        for prefix in ('all', 'extra'):
            volume = np.load(one_file(input_root, prefix+'_vols.npy'), mmap_mode='r', allow_pickle=False)
            masks = np.load(one_file(input_root, prefix+'_masks.npy'), allow_pickle=False)
            if volume.shape[1:] != (44,336,336) or masks.shape != volume.shape[:2]:
                raise ValueError('unexpected corpus layout')
            parts.append((volume, masks))
        entries = [(p,r,int((mask > 0).sum())) for p,(_,masks) in enumerate(parts) for r,mask in enumerate(masks)]
        if any(n == 0 for _,_,n in entries):
            raise ValueError('empty corpus studies require explicit audit')
        entries.sort(key=lambda x: (x[2], x[0], x[1]))
        chosen = [entries[i] for i in np.linspace(0,len(entries)-1,96).round().astype(int)]
        np.random.default_rng(20260905).shuffle(chosen)
        receipt['sample_rows'] = [{'part':p,'row':r,'valid_slices':n} for p,r,n in chosen]
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            model(profiling_windows(parts[chosen[0][0]][0][chosen[0][1]], parts[chosen[0][0]][1][chosen[0][1]],24)[None].cuda())
        torch.cuda.synchronize()
        receipt['fixed_seconds'] = time.perf_counter() - started
        times, data_times, forward_times = [], [], []
        with torch.inference_mode():
            for p,r,_ in chosen:
                begin = time.perf_counter()
                x = profiling_windows(parts[p][0][r], parts[p][1][r],24)[None].cuda()
                torch.cuda.synchronize()
                loaded = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.float16):
                    prediction = model(x).float()
                if not torch.isfinite(prediction).all():
                    raise RuntimeError('nonfinite forward output')
                torch.cuda.synchronize()
                ended = time.perf_counter()
                times.append(ended-begin)
                data_times.append(loaded-begin)
                forward_times.append(ended-loaded)
                if len(times) in (48,96):
                    print(json.dumps({'profile_studies':len(times), 'mean_seconds':float(np.mean(times))}), flush=True)
        receipt['study_seconds'] = times
        receipt['data_seconds'] = data_times
        receipt['forward_seconds'] = forward_times
        receipt['compute_only_projection'] = runtime_upper_bound(receipt['fixed_seconds'], times)
        receipt['hidden_end_to_end_projection'] = None  # DICOM costs not measured
        model.train()
        freeze_batchnorm_statistics(model)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=.01)
        scaler = torch.amp.GradScaler('cuda')
        train_times = []
        optimizer.zero_grad(set_to_none=True)
        for step in range(40):  # 8 warm-up + 32 measured, including optimizer steps
            begin = time.perf_counter()
            p,r,_ = chosen[step]
            x = profiling_windows(parts[p][0][r], parts[p][1][r],12)[None].cuda()
            # Synthetic targets for compute only. No checkpoints/predictions are
            # exported and this model must never become a competition candidate.
            target = torch.full((1,12), .5, device='cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                loss = soft_weighted_bce(model(x), target, torch.ones_like(target)) / 8
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite profiling loss')
            scaler.scale(loss).backward()
            if (step+1) % 8 == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params,1.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if step >= 8:
                train_times.append(time.perf_counter()-begin)
        receipt['train_study_seconds'] = train_times
        train_rate, eval_rate = float(np.mean(train_times)), float(np.mean(times))
        # Largest train/inner/outer partitions give a conservative one-fold scenario.
        seconds = 3*(2321*train_rate + 620*eval_rate) + 1451*eval_rate
        receipt['one_fold_training_upper_seconds'] = receipt['fixed_seconds'] + seconds*1.2
        receipt['three_fold_gpu_hours_scenario'] = 3*receipt['one_fold_training_upper_seconds']/3600
        receipt['pilot_compute_feasible'] = receipt['one_fold_training_upper_seconds'] <= 4.25*3600
        receipt['status'] = 'complete'
    except Exception as error:
        receipt['status'] = 'failed'
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        receipt['elapsed_seconds'] = time.perf_counter()-started
        receipt['peak_vram_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        (output/'runtime_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(json.dumps(receipt),flush=True)
    return receipt
