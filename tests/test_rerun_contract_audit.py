import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from rsna_knee.constants import LABELS
from rsna_knee.rerun_contract_audit import audit_rerun_contract


def synthetic_contract_fixture(output, native, code):
    reference = pd.read_csv(output / 'submission.csv', dtype={'StudyInstanceUID':str})
    cases=[]
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    for count in (1,2,3):
        root=output.parent/'rerun_contract'/str(count)/'native_reader_inference';root.mkdir(parents=True)
        reference.iloc[:count].to_csv(root/'submission.csv',index=False)
        receipt={**native,'execution_mode':'competition_rerun','training_timing_study_count':0,'test_study_count':count,'runner_sha256':code}
        (root/'receipt.json').write_text(json.dumps(receipt))
        cases.append(dict(test_count=count,actual_training_studies=0,test_only_mount=True,training_access_attempts=0,
            dataframe_read_paths=['test.csv','test_series.csv'],maximum_probability_difference=0.,
            receipt_sha256=sha(root/'receipt.json'),csv_sha256=sha(root/'submission.csv')))
    report=dict(status='COMPLETE',actual_hidden_data_accessed=False,actual_public_mri_predictions=6,cases=cases,
        scope='SYNTHETIC UNIT TEST ONLY; never actual deployment evidence')
    (output.parent/'rerun_contract/receipt.json').write_text(json.dumps(report))
    return report


class RerunContractAuditTests(unittest.TestCase):
    def test_csv_is_recomputed_even_if_report_claim_and_updated_hash_agree(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'native_reader_inference';output.mkdir()
            frame=pd.DataFrame([{**{'StudyInstanceUID':f'u{row}'},**{label:.2+row*.1 for label in LABELS}} for row in range(3)])
            frame.to_csv(output/'submission.csv',index=False)
            native=dict(status='COMPLETE',decode_fallback_count=0,missing_study_count=0,test_reports_read=False)
            report=synthetic_contract_fixture(output,native,'a'*64)
            audit_rerun_contract(output,dict(code_sha256='a'*64))
            path=output.parent/'rerun_contract/1/native_reader_inference/submission.csv'
            changed=pd.read_csv(path);changed.loc[0,'ACL']+=.01;changed.to_csv(path,index=False)
            report['cases'][0]['csv_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            (output.parent/'rerun_contract/receipt.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'predictions differ'):
                audit_rerun_contract(output,dict(code_sha256='a'*64))


if __name__=='__main__':unittest.main()
