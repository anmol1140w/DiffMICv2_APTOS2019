import tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from aptos_research.gate import evaluate_gate, qrc_is_eligible
from aptos_research.features import FeatureStore
from aptos_research.checkpoint import inspect_state_dict

class ResearchTests(unittest.TestCase):
 def test_gate_decisions_and_artifact(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'gate.json'
   inc=evaluate_gate({},out,bootstrap_samples=200);self.assertEqual(inc['decision'],'INCONCLUSIVE')
   labels=[0]*100
   passed=evaluate_gate({'validation':{'ids':['x%d'%i for i in range(100)],'labels':labels,'ordered_predictions':labels,'control_predictions':[[1]*40+[0]*60]}},out,minimum_delta=.05,bootstrap_samples=1000,seed=1)
   self.assertEqual(passed['decision'],'PASS');self.assertTrue(qrc_is_eligible(out))
   failed=evaluate_gate({'validation':{'ids':['x%d'%i for i in range(100)],'labels':labels,'ordered_predictions':labels,'control_predictions':[labels]}},out,bootstrap_samples=100,seed=1)
   self.assertEqual(failed['decision'],'FAIL');self.assertFalse(qrc_is_eligible(out))
 def test_feature_store_schema_and_empty_integrity(self):
  with tempfile.TemporaryDirectory() as d:
   store=FeatureStore(str(Path(d)/'features.sqlite'));self.assertEqual(store.integrity()['splits'],{});store.close()
 def test_checkpoint_compatibility_reports_shapes_and_keys(self):
  target={'a':SimpleNamespace(shape=(2,3)),'b':SimpleNamespace(shape=(5,))}
  incoming={'a':SimpleNamespace(shape=(2,3)),'b':SimpleNamespace(shape=(4,)),'unused':SimpleNamespace(shape=(1,))}
  result=inspect_state_dict(target,incoming)['report']
  self.assertEqual(result['loaded_keys'],['a']);self.assertIn('b',result['mismatched_shapes']);self.assertEqual(result['unexpected_keys'],['unused'])

if __name__=='__main__':unittest.main()
