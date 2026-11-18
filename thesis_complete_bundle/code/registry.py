"""Append-only JSONL experiment event log."""
import json,time
from pathlib import Path

def record(path, stage, status, config, artifacts=None, metrics=None, note=''):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    event={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':stage,'status':status,'config':config,'artifacts':artifacts or [],'metrics':metrics or {},'note':note}
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(event,sort_keys=True)+'\n')
    return event
