"""Resumable, split-aware SQLite trajectory cache (float32 vector BLOBs)."""
import hashlib, json, sqlite3, time
from pathlib import Path

class FeatureStore:
    def __init__(self, path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(str(self.path))
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS states (sample_id TEXT NOT NULL, split TEXT NOT NULL, label INTEGER NOT NULL, state_index INTEGER NOT NULL, diffusion_timestep INTEGER NOT NULL, dimension INTEGER NOT NULL, representation BLOB NOT NULL, checkpoint_id TEXT NOT NULL, config_json TEXT NOT NULL, seed INTEGER NOT NULL, created REAL NOT NULL, PRIMARY KEY(sample_id, split, state_index))''')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        version=self.db.execute('SELECT value FROM metadata WHERE key="schema_version"').fetchone()
        if version and int(version[0])!=1: raise ValueError('Unsupported trajectory cache schema version: %s'%version[0])
        self.db.execute('INSERT OR REPLACE INTO metadata(key,value) VALUES("schema_version","1")')
        self.db.execute('CREATE INDEX IF NOT EXISTS states_split_idx ON states(split, sample_id)')
        self.db.commit()
    def add_sample(self, sample_id, split, label, states, timesteps, checkpoint_id, config, seed):
        import numpy as np
        if split not in ('train','validation','test'): raise ValueError('invalid split')
        states=np.asarray(states,dtype=np.float32)
        if states.ndim!=2 or not np.isfinite(states).all(): raise ValueError('states must be finite [K,D]')
        if states.shape[0]!=len(timesteps): raise ValueError('one original timestep required per state')
        if int(label) not in range(5): raise ValueError('APTOS grade must be 0..4')
        cfg=json.dumps(config,sort_keys=True)
        with self.db:
            for i,(state,t) in enumerate(zip(states,timesteps)):
                self.db.execute('INSERT OR REPLACE INTO states VALUES(?,?,?,?,?,?,?,?,?,?,?)',(str(sample_id),split,int(label),i,int(t),states.shape[1],sqlite3.Binary(state.tobytes()),str(checkpoint_id),cfg,int(seed),time.time()))
    def contains(self, sample_id, split, label=None, checkpoint_id=None, config=None, seed=None):
        row=self.db.execute('SELECT label,checkpoint_id,config_json,seed FROM states WHERE sample_id=? AND split=? LIMIT 1',(str(sample_id),split)).fetchone()
        if row is None: return False
        old_label,old_checkpoint,old_config,old_seed=row
        if label is not None and old_label!=int(label): raise ValueError('Cached label mismatch for %s'%sample_id)
        if checkpoint_id is not None and old_checkpoint!=str(checkpoint_id): raise ValueError('Cached checkpoint mismatch for %s; choose a new cache path'%sample_id)
        if seed is not None and old_seed!=int(seed): raise ValueError('Cached extraction seed mismatch for %s; choose a new cache path'%sample_id)
        if config is not None:
            old=json.loads(old_config); new=dict(config)
            old.pop('input_state_shape',None);new.pop('input_state_shape',None)
            if old!=new: raise ValueError('Cached extraction configuration mismatch for %s; choose a new cache path'%sample_id)
        return True
    def load_split(self, split, sample_ids=None, limit=None):
        import numpy as np
        query='SELECT sample_id,label,state_index,diffusion_timestep,dimension,representation,checkpoint_id,config_json,seed FROM states WHERE split=?'; params=[split]
        if sample_ids is not None:
            ids=list(sample_ids)
            if not ids: return []
            query+=' AND sample_id IN (%s)'%','.join('?' for _ in ids);params.extend(ids)
        if limit is not None:
            if int(limit)<1: return []
            query+=' AND sample_id IN (SELECT sample_id FROM states WHERE split=? ORDER BY sample_id LIMIT ?)';params.extend([split,int(limit)])
        query+=' ORDER BY sample_id,state_index'
        rows=self.db.execute(query,params).fetchall()
        grouped={}
        for sid,label,idx,t,d,blob,ckpt,cfg,seed in rows:
            entry=grouped.setdefault(sid,{'label':label,'states':[],'timesteps':[],'checkpoint_id':ckpt,'config':json.loads(cfg),'seed':seed})
            if idx!=len(entry['states']): raise ValueError('cache state indices are not contiguous for '+sid)
            entry['states'].append(np.frombuffer(blob,dtype=np.float32,count=d).copy()); entry['timesteps'].append(t)
        return [{'id':sid,**v,'states':np.stack(v['states'])} for sid,v in grouped.items()]
    def integrity(self):
        splits={}
        for split, in self.db.execute('SELECT DISTINCT split FROM states'):
            samples=self.db.execute('SELECT COUNT(DISTINCT sample_id) FROM states WHERE split=?',(split,)).fetchone()[0]
            dims=self.db.execute('SELECT COUNT(DISTINCT dimension) FROM states WHERE split=?',(split,)).fetchone()[0]
            splits[split]={'samples':samples,'dimensions':dims,'state_rows':self.db.execute('SELECT COUNT(*) FROM states WHERE split=?',(split,)).fetchone()[0]}
        issues=[]
        if not self.db.execute('SELECT 1 FROM states LIMIT 1').fetchone(): issues.append('feature cache is empty')
        if self.db.execute('SELECT COUNT(*) FROM states WHERE label<0 OR label>4 OR dimension<1 OR split NOT IN ("train","validation","test")').fetchone()[0]: issues.append('invalid label/dimension/split')
        for sid,split,n,min_i,max_i,dim_count,label_count in self.db.execute('SELECT sample_id,split,COUNT(*),MIN(state_index),MAX(state_index),COUNT(DISTINCT dimension),COUNT(DISTINCT label) FROM states GROUP BY sample_id,split'):
            if min_i!=0 or max_i!=n-1: issues.append('non-contiguous state indices for %s/%s'%(sid,split))
            if dim_count!=1 or label_count!=1: issues.append('inconsistent dimension or label for %s/%s'%(sid,split))
        if self.db.execute('SELECT COUNT(*) FROM (SELECT sample_id FROM states GROUP BY sample_id HAVING COUNT(DISTINCT split)>1)').fetchone()[0]: issues.append('sample IDs overlap across splits')
        state_count_variants=self.db.execute('SELECT COUNT(DISTINCT n) FROM (SELECT sample_id,split,COUNT(*) AS n FROM states GROUP BY sample_id,split)').fetchone()[0]
        if state_count_variants>1: issues.append('inconsistent trajectory lengths across samples')
        digest=None
        if self.path.exists():
            self.db.execute('PRAGMA wal_checkpoint(FULL)')
            h=hashlib.sha256()
            with self.path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
            digest=h.hexdigest()
        return {'schema_version':1,'path':str(self.path.resolve()),'splits':splits,'valid':not issues,'issues':sorted(set(issues)),'sha256':digest}
    def close(self): self.db.close()
