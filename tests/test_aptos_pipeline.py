import csv, tempfile, unittest
from pathlib import Path
from PIL import Image
from aptos_research.data import DatasetError, resolve_aptos, make_splits, save_split_manifest, load_split_manifest
from aptos_research.metrics import classification_metrics

class AptosTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'train_images').mkdir()
  with (self.root/'train.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['id_code','diagnosis'])
   for i in range(20):
    w.writerow(['img%02d'%i,i%5]);Image.new('RGB',(20+i,30),(i,2,3)).save(self.root/'train_images'/('IMG%02d.JPG'%i))
 def tearDown(self):self.tmp.cleanup()
 def test_csv_ids_case_insensitive_and_distribution(self):
  records,stats=resolve_aptos(str(self.root));self.assertEqual(len(records),20);self.assertEqual(stats['class_counts'],{'0':4,'1':4,'2':4,'3':4,'4':4})
 def test_splits_are_disjoint_and_deterministic(self):
  records,_=resolve_aptos(str(self.root));a=make_splits(records,seed=7);b=make_splits(records,seed=7);self.assertEqual(a,b)
  ids=[{x['id'] for x in a[k]} for k in ('train','validation','test')];self.assertFalse(ids[0]&ids[1]);self.assertFalse(ids[0]&ids[2]);self.assertFalse(ids[1]&ids[2])
  manifest=self.root/'splits.json';save_split_manifest(a,manifest,7);self.assertEqual(load_split_manifest(manifest)['seed'],7)
 def test_invalid_label_and_missing_image_fail(self):
  p=self.root/'train.csv';p.write_text('id_code,diagnosis\nimg00,8\n',encoding='utf-8')
  with self.assertRaises(DatasetError):resolve_aptos(str(self.root))
  p.write_text('id_code,diagnosis\nabsent,1\n',encoding='utf-8')
  with self.assertRaises(DatasetError):resolve_aptos(str(self.root))
 def test_multiclass_metrics_include_ordinal_and_calibration(self):
  result=classification_metrics([0,1,2,3,4],[[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
  self.assertEqual(result['accuracy'],1);self.assertEqual(result['quadratic_weighted_kappa'],1);self.assertEqual(result['ordinal_mae'],0);self.assertEqual(result['macro_auroc_ovr'],1);self.assertEqual(result['ece_10_bins'],0)

if __name__=='__main__':unittest.main()
