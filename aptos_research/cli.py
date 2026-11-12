"""Command-line entry points for indexing and trajectory experiments."""
import argparse, json, sys
from pathlib import Path
from aptos_research.data import resolve_aptos,make_splits,save_split_manifest,load_split_manifest
from aptos_research.features import FeatureStore
from aptos_research.gate import evaluate_gate
from aptos_research.registry import record

def validate(args):
    records,stats=resolve_aptos(args.root,args.csv,args.images)
    corrupt=[]
    try:
        from PIL import Image
        for r in records:
            try:
                with Image.open(r['path']) as im: im.verify()
            except Exception as exc: corrupt.append({'id':r['id'],'path':r['path'],'error':str(exc)})
    except ImportError: print('WARNING: Pillow unavailable; image corruption scan not run',file=sys.stderr)
    if corrupt: raise RuntimeError('%d corrupt/unreadable images; first=%s' % (len(corrupt, ),corrupt[0]))
    splits=make_splits(records,args.seed,args.validation_fraction,args.test_fraction)
    save_split_manifest(splits,args.output,args.seed,stats)
    record('outputs/experiments/registry.jsonl','dataset-validation','COMPLETED',{'root':stats['root'],'csv':stats['csv'],'seed':args.seed,'split_manifest':str(Path(args.output).resolve())},{'split_manifest':str(Path(args.output).resolve())},stats)
    print(json.dumps({'status':'validated','stats':stats,'split_counts':{k:len(v) for k,v in splits.items()},'manifest':str(Path(args.output).resolve())},indent=2))

def run_gate(args):
    results=json.loads(Path(args.results).read_text())
    result=evaluate_gate(results,args.output,args.minimum_delta,args.alpha,args.bootstrap_samples,args.seed)
    record('outputs/experiments/registry.jsonl','classical-gate',result['decision'],{'minimum_delta':args.minimum_delta,'alpha':args.alpha,'bootstrap_samples':args.bootstrap_samples,'seed':args.seed},{'gate':str(Path(args.output).resolve())},result)
    print(json.dumps(result,indent=2))
    if result['decision']!='PASS': return 2 if result['decision']=='FAIL' else 3
    return 0

def main():
    p=argparse.ArgumentParser(prog='aptos-research'); sub=p.add_subparsers(dest='command',required=True)
    v=sub.add_parser('validate-data',help='validate APTOS CSV/images and write stratified split manifest')
    v.add_argument('--root',required=True);v.add_argument('--csv');v.add_argument('--images');v.add_argument('--output',default='outputs/aptos_splits.json');v.add_argument('--seed',type=int,default=2026);v.add_argument('--validation-fraction',type=float,default=.15);v.add_argument('--test-fraction',type=float,default=.15);v.set_defaults(func=validate)
    i=sub.add_parser('inspect-cache');i.add_argument('cache');i.set_defaults(func=lambda a:print(json.dumps(FeatureStore(a.cache).integrity(),indent=2)))
    g=sub.add_parser('gate');g.add_argument('--results',required=True);g.add_argument('--output',default='outputs/experiments/gate.json');g.add_argument('--minimum-delta',type=float,default=.01);g.add_argument('--alpha',type=float,default=.05);g.add_argument('--bootstrap-samples',type=int,default=5000);g.add_argument('--seed',type=int,default=2026);g.set_defaults(func=run_gate)
    args=p.parse_args()
    try:
        result=args.func(args)
        return result if isinstance(result,int) else 0
    except Exception as exc:
        print('ERROR: %s' % exc,file=sys.stderr)
        return 2
if __name__=='__main__': raise SystemExit(main())
