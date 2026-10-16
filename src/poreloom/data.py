"""Read-only access to existing short-window and encoder-feature caches."""
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import numpy as np
from .contracts import ReadRecord
from .windowing import BOUNDS

@dataclass
class ControlDataset:
    features: object
    valid: np.ndarray
    is_target: np.ndarray
    raw_length: np.ndarray
    read_ids: np.ndarray
    windows: object
    split: str
    run_or_batch: np.ndarray
    sample_rate_hz: float=4000.
    def __len__(self):return len(self.read_ids)
    @property
    def labels(self):return self.is_target.astype(np.int64)
    @property
    def complete(self):return self.valid.all(1)
    @property
    def max_windows(self):return self.valid.shape[1]
    def record(self,i):
        return ReadRecord(str(self.read_ids[i]),bool(self.is_target[i]),int(self.raw_length[i]),self.valid[i],None if self.windows is None else self.windows[i],None if self.features is None else self.features[i],self.sample_rate_hz,self.split,str(self.run_or_batch[i]))
    def records(self,indices=None):return [self.record(int(i)) for i in (range(len(self)) if indices is None else indices)]

def load_dataset(data_dir,split,max_windows=4,features_path=None,sample_rate_hz=4000.):
    root=Path(data_dir);folder=root/split
    if not folder.exists() and root.name==split:folder=root
    metadata=np.load(folder/'metadata.npz',allow_pickle=False)
    valid=np.load(folder/'valid.npy',mmap_mode='r')[:,:max_windows]
    if max_windows<1 or max_windows>valid.shape[1]:raise ValueError('invalid window budget')
    path=Path(features_path) if features_path else folder.parent.parent/f'{split}_features.npy'
    features=np.load(path,mmap_mode='r')[:,:max_windows] if path.exists() else None
    windows=np.load(folder/'windows.npy',mmap_mode='r')[:,:max_windows]
    labels=metadata['network_labels']==1 if 'network_labels' in metadata else metadata['labels']==0
    # Source chunks identify batch provenance, never independent biological runs.
    provenance=metadata['run_or_batch'] if 'run_or_batch' in metadata else np.array([str(Path(x).parent) for x in metadata['source_chunk']]) if 'source_chunk' in metadata else np.full(len(labels),'unknown')
    return ControlDataset(features,valid,labels,metadata['raw_length'],metadata['read_id'],windows,split,provenance,sample_rate_hz)

def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as source:
        for chunk in iter(lambda:source.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def prepare_data(data_dir,output,max_windows=4,sample_rate_hz=4000.,**kwargs):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    arrays={k:[] for k in ('read_id','split','run_or_batch','is_target','raw_length','sample_rate_hz','valid')}
    manifests={}
    for split in ('train','dev','test'):
        if not (Path(data_dir)/split).exists():continue
        d=load_dataset(data_dir,split,max_windows,sample_rate_hz=sample_rate_hz)
        for name in arrays:
            arrays[name].append(np.full(len(d),split) if name=='split' else np.full(len(d),sample_rate_hz) if name=='sample_rate_hz' else d.read_ids if name=='read_id' else getattr(d,name))
        manifest=Path(data_dir)/split/'manifest.json'
        if manifest.exists():
            manifests[split]=json.loads(manifest.read_text())
            for filename,expected in manifests[split].get('files',{}).items():
                if file_sha256(manifest.parent/filename)!=expected:raise ValueError(f'cache hash mismatch: {split}/{filename}')
            manifests[split]['verified_cache_files']=True
        feature_path=Path(data_dir).parent/f'{split}_features.npy'
        if feature_path.exists():manifests.setdefault(split,{})['feature_cache']={'path':str(feature_path.resolve()),'sha256':file_sha256(feature_path)}
    if not arrays['read_id']:raise ValueError('no cached splits')
    np.savez_compressed(output/'read_index.npz',**{k:np.concatenate(v) for k,v in arrays.items()},window_bounds=BOUNDS[:max_windows])
    (output/'data_manifest.json').write_text(json.dumps({'data_dir':str(Path(data_dir).resolve()),'max_windows':max_windows,'sample_rate_hz':sample_rate_hz,'cache_manifests':manifests,'provenance_note':'source batch paths are not independent experimental runs'},indent=2))
    return output/'read_index.npz'
