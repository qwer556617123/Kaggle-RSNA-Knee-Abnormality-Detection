from __future__ import annotations

LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

CONFIDENCE_COLUMNS = [f"{label}__confidence" for label in LABELS]
LOSS_MASK_COLUMNS = [f"{label}__loss_mask" for label in LABELS]
LABEL_VERSION = "multilingual-v1"

# A stable order is part of the model artifact contract.
SLOT_NAMES = (
    "Axial_Fluid",
    "Sagittal_Fluid",
    "Coronal_Fluid",
    "Sagittal_NonFluid",
    "Coronal_NonFluid",
)

OFFICIAL_TEST_STUDIES = 1300
MAX_NOTEBOOK_SECONDS = 9 * 60 * 60
INFERENCE_TARGET_SECONDS = int(MAX_NOTEBOOK_SECONDS * 0.85)
