"""Compare complete research OOF bundles; never queries or fits public LB scores."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rsna_knee.research_coatnet import load_contract
from rsna_knee.research_oof import compare_oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract',required=True,type=Path)
    parser.add_argument('--control',required=True,type=Path)
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    frame,_,contract = load_contract(args.contract)
    runs = [json.loads((p/'run_manifest.json').read_text()) for p in (args.control,args.candidate)]
    predictions = [pd.read_csv(p/'oof.csv',dtype={'StudyInstanceUID':str}) for p in (args.control,args.candidate)]
    report = compare_oof(frame,*predictions,*runs,contract)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
