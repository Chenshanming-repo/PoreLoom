"""Causal ReadCurrent ADC preprocessing and contiguous, nonoverlapping windows."""
import numpy as np
START, WINDOW, MAX_WINDOWS = 1500, 1500, 8
BOUNDS = np.array([(START+k*WINDOW, START+(k+1)*WINDOW) for k in range(MAX_WINDOWS)])


def normalize_windows(raw):
    x=np.asarray(raw)
    if x.ndim!=2 or x.shape[1]!=WINDOW:raise ValueError('expected [N,1500] windows')
    x=x.astype(np.float64);finite=np.isfinite(x).all(1)
    z=np.zeros_like(x);median=np.zeros(len(x));mad=np.zeros(len(x))
    median[finite]=np.median(x[finite],axis=1)
    dev=x-median[:,None];mad[finite]=np.median(np.abs(dev[finite]),axis=1)
    valid=finite&(mad>1e-8)
    z[valid]=dev[valid]/(1.4826*mad[valid,None])
    for row in np.flatnonzero(valid):
        for idx in np.flatnonzero(np.abs(z[row])>3.5):
            if idx==0:z[row,idx]=z[row,1]
            elif idx==WINDOW-1:z[row,idx]=z[row,idx-1]
            else:z[row,idx]=(z[row,idx-1]+z[row,idx+1])/2
    return z.astype(np.float32),valid


def split_windows(raw):
    raw=np.asarray(raw)
    if raw.ndim!=1:raise ValueError('expected one-dimensional ADC')
    out=np.zeros((MAX_WINDOWS,WINDOW),dtype=raw.dtype);valid=np.zeros(MAX_WINDOWS,bool)
    for k,(start,end) in enumerate(BOUNDS):
        if len(raw)>=end:out[k]=raw[start:end];valid[k]=True
    return out,valid


class SignalBuffer:
    """Offsets are absolute read sample indices. Reject gaps; drop repeated samples."""
    def __init__(self,max_windows=8):
        if not 1 <= max_windows <= 8: raise ValueError("max_windows must be in [1,8]")
        self.max_windows=max_windows
        self.received=0;self.next_start=START;self.buffer=np.empty(0,np.float64);self.origin=0
    def push(self,offset,signal):
        signal=np.asarray(signal)
        if signal.ndim!=1 or offset<0:raise ValueError('invalid signal chunk')
        if offset>self.received:raise ValueError('noncontiguous signal offset')
        end=offset+len(signal)
        if end<=self.received:return
        fresh=signal[self.received-offset:];self.buffer=np.concatenate([self.buffer,fresh]);self.received=end
        discard=max(0,min(len(self.buffer),self.next_start-self.origin))
        self.buffer=self.buffer[discard:];self.origin+=discard
    def pop(self):
        if self.next_start>=START+self.max_windows*WINDOW:return None
        end=self.next_start+WINDOW
        if self.received<end:return None
        offset=self.next_start-self.origin;out=self.buffer[offset:offset+WINDOW].copy()
        self.next_start=end;drop=end-self.origin;self.buffer=self.buffer[drop:];self.origin=end
        return out
