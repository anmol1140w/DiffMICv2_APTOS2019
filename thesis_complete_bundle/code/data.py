"""Strict APTOS 2019 indexing, integrity checks, and reproducible split metadata."""
from __future__ import annotations
import csv, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

class DatasetError(ValueError):
    pass

def _identifier(value: str) -> str:
    value = Path(value.strip()).name
    return Path(value).stem.casefold()

def resolve_aptos(root: str, csv_path: Optional[str] = None, image_dir: Optional[str] = None):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise DatasetError('APTOS root does not exist or is not a directory: %s' % root)
    csv_candidates = [Path(csv_path).expanduser() if csv_path else root/'train.csv', root/'labels.csv', root/'trainLabels.csv']
    csv_file = next((p.resolve() for p in csv_candidates if p.is_file()), None)
    if csv_file is None:
        raise DatasetError('Could not find annotation CSV. Pass --csv or place train.csv under %s' % root)
    image_roots = [Path(image_dir).expanduser() if image_dir else root/'train_images', root/'images', root]
    image_roots = [p.resolve() for p in image_roots if p.is_dir()]
    if not image_roots:
        raise DatasetError('No image directory found; pass --images or provide train_images/')
    image_index: Dict[str, List[Path]] = defaultdict(list)
    for directory in image_roots:
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                resolved=path.resolve()
                if resolved not in image_index[_identifier(path.name)]:
                    image_index[_identifier(path.name)].append(resolved)
    records, seen = [], set()
    duplicate_ids, missing, invalid, duplicate_files = [], [], [], []
    with csv_file.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise DatasetError('Annotation CSV has no header: %s' % csv_file)
        fields = {x.strip().casefold(): x for x in reader.fieldnames}
        id_key = next((fields[k] for k in ('id_code','image','image_id','filename','id') if k in fields), None)
        label_key = next((fields[k] for k in ('diagnosis','label','grade','level') if k in fields), None)
        if id_key is None or label_key is None:
            raise DatasetError('Expected image id and diagnosis columns; found %s' % reader.fieldnames)
        for line, row in enumerate(reader, 2):
            ident = _identifier(row.get(id_key, ''))
            if not ident:
                invalid.append({'line': line, 'reason': 'empty identifier'}); continue
            if ident in seen:
                duplicate_ids.append(ident); continue
            seen.add(ident)
            try:
                label = int(row[label_key])
                if label not in range(5): raise ValueError()
            except (TypeError, ValueError):
                invalid.append({'line': line, 'id': ident, 'label': row.get(label_key)}); continue
            matches = image_index.get(ident, [])
            if not matches:
                missing.append(ident); continue
            if len(matches) > 1:
                duplicate_files.append({'id': ident, 'paths': [str(x) for x in matches]}); continue
            records.append({'id': ident, 'label': label, 'path': str(matches[0])})
    if duplicate_ids or missing or invalid or duplicate_files:
        raise DatasetError('APTOS integrity errors: %d duplicate annotation IDs, %d missing images, %d invalid rows, %d ambiguous image IDs. Use --report to inspect all details.' % (len(duplicate_ids),len(missing),len(invalid),len(duplicate_files)))
    if not records:
        raise DatasetError('APTOS index is empty; refusing to continue.')
    return records, {'root': str(root), 'csv': str(csv_file), 'image_roots': [str(x) for x in image_roots], 'samples': len(records), 'class_counts': {str(k):v for k,v in sorted(Counter(r['label'] for r in records).items())}, 'identifier_sha256': hashlib.sha256('\n'.join(sorted(r['id'] for r in records)).encode()).hexdigest()}

def make_splits(records: Sequence[dict], seed: int = 2026, validation_fraction: float = .15, test_fraction: float = .15):
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError('Validation/test fractions must be positive and sum to less than one')
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for rec in records: by_class[int(rec['label'])].append(rec)
    result = {'train': [], 'validation': [], 'test': []}
    for label, values in sorted(by_class.items()):
        values = list(values); rng.shuffle(values); n = len(values)
        nt = max(1, round(n*test_fraction)) if n >= 3 else 0
        nv = max(1, round(n*validation_fraction)) if n >= 2 else 0
        while nt + nv >= n and (nt or nv):
            if nt >= nv and nt: nt -= 1
            elif nv: nv -= 1
        result['test'].extend(values[:nt]); result['validation'].extend(values[nt:nt+nv]); result['train'].extend(values[nt+nv:])
    for split in result.values(): rng.shuffle(split)
    ids = [{x['id'] for x in result[k]} for k in ('train','validation','test')]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i+1,3)):
        raise AssertionError('Internal split overlap')
    return result

def save_split_manifest(splits, path, seed, stats=None):
    payload = {'schema_version': 1, 'seed': int(seed), 'stats': stats or {}, 'splits': {k: [{'id':r['id'],'label':int(r['label']),'path':r['path']} for r in v] for k,v in splits.items()}}
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    return payload

def load_split_manifest(path):
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    splits=payload.get('splits')
    if not isinstance(splits,dict) or not {'train','validation','test'} <= set(splits): raise DatasetError('Invalid split manifest: require train/validation/test')
    sets=[]
    for name in ('train','validation','test'):
        ids=set()
        for r in splits[name]:
            if int(r['label']) not in range(5): raise DatasetError('Invalid grade for %s in %s' % (r.get('id'),name))
            if r['id'] in ids: raise DatasetError('Duplicate ID %s in %s' % (r['id'],name))
            ids.add(r['id'])
        sets.append(ids)
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)): raise DatasetError('Split manifest overlaps across train/validation/test')
    return payload

class AptosImageDataset:
    """Torch Dataset loaded lazily. Returns image, grade, identifier, split."""
    def __init__(self, records, split, size=224, training=False):
        if split not in ('train','validation','test'): raise ValueError(split)
        self.records=list(records); self.split=split; self.size=int(size); self.training=bool(training and split=='train')
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        try:
            import torch
            from PIL import Image, ImageOps
        except ImportError as exc: raise RuntimeError('Install torch, torchvision, and Pillow to load model batches') from exc
        rec=self.records[index]
        with Image.open(rec['path']) as image:
            image=ImageOps.exif_transpose(image).convert('RGB')
            w,h=image.size; side=min(w,h); left=(w-side)//2; top=(h-side)//2
            image=image.crop((left,top,left+side,top+side)).resize((self.size,self.size),Image.Resampling.BILINEAR)
            if self.training:
                import random
                if random.random()<.5: image=ImageOps.mirror(image)
                if random.random()<.5: image=ImageOps.flip(image)
                image=image.rotate(random.uniform(-30,30),resample=Image.Resampling.BILINEAR)
            import numpy as np
            arr=np.asarray(image,dtype='float32')/255.0
        tensor=torch.from_numpy(arr).permute(2,0,1)
        mean=torch.tensor([.485,.456,.406])[:,None,None]; std=torch.tensor([.229,.224,.225])[:,None,None]
        return (tensor-mean)/std, torch.tensor(int(rec['label']),dtype=torch.long), rec['id'], self.split
