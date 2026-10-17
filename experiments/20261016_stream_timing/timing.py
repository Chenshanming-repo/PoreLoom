#!/usr/bin/env python3
"""Incremental (StreamR0Bank) vs from-scratch StreamR0 compute time per chunk, as a function of the signal already
received.

Model: the budget-18000 S1 encoder of r10_run01 (runs/budget_up/r10_run01/b18000_r0/stream/best.pt; StreamR0,
max_samples 19500, 4 kHz) so the stream can run to 4.9 s. Reads: r10_run01 TEST reads with raw_length >= 19500 from the
27000-sample cache, pA from per-read calibration; disjoint reads per batch and repetition (seed 20261016).

One decision update = what the live client does when a chunk of C samples arrives for B reads at once:
  incremental  StreamR0Bank.step(new chunk of every read) + readout -> logits on the CPU. The bank keeps each read's
               normalized stream, block statistics, the sum of final frames and the tail; per update it runs the
               original layers once over [48F - 528, used) per read (528-sample context + new samples).
  scratch      the client keeps the raw signal and recomputes R0 from the read start: upload pA [skip, skip + L'),
               causal block normalization, the original (two-sided padding) backbone over the whole stream of every
               read in one ragged pass (stream_encoder._ragged), mean over all frames, projection + head + time term
               -> logits on the CPU. This is R0's definition, so both give the same logits (checked every step).
Both include the host -> GPU upload and the logits' copy back (torch.cuda.synchronize before / after).
Scenarios:
  aligned  B reads start together; every chunk adds C samples to all; the update after each chunk is timed, so the
           time is a function of the signal received n (C, 2C, ... until the 19500 cap).
  mixed    the live steady state: each of B reads is at its own random received length n ~ U[max(C, 1932), 19500];
           its state up to n - C is built untimed in one step, then one chunk update of C samples is timed.
Grid: chunk C in {800, 1600, 4000, 8000, 12000, 18000} samples (0.2 / 0.4 / 1 / 2 / 3 / 4.5 s at 4 kHz), batch B in {1, 16, 64, 256, 512},
3 repetitions (median reported) after one untimed warm-up repetition per (C, B).

    timing.py run    --data <r10_run01_p27000> --model <b18000_r0/stream/best.pt> --out <dir> [--device cuda:0]
    timing.py report --out <dir>   (summary_zh.md)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'src'))
from poreloom.prefix_data import load_prefix_cache  # noqa: E402
from poreloom.stream_bank import StreamR0Bank  # noqa: E402
from poreloom.stream_encoder import _ragged  # noqa: E402
from poreloom.stream_training import load_stream_r0  # noqa: E402

CHUNKS = (800, 1600, 4000, 8000, 12000, 18000)
BATCHES = (1, 16, 64, 256, 512)
REPS = 3
SEED = 20261016
FS = 4000.


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


@torch.no_grad()
def scratch(model, pa, n, device):
    """R0 logits [B,2] (CPU) recomputed from the read start for reads holding n[i] samples (pa: [B, >= max(n)] pA)."""
    m = model
    used = m.used(np.clip(n - m.skip, 0, m.max_samples - m.skip))
    width = int(used.max())
    x = torch.from_numpy(np.ascontiguousarray(pa[:, m.skip:m.skip + width], np.float32))
    if device.type == 'cuda': x = x.pin_memory().to(device, non_blocking=True)
    u = torch.from_numpy(used).to(device, non_blocking=True)
    past = torch.arange(width, device=device) >= u[:, None]
    z = m.normalizer(x.masked_fill(past, 0)).masked_fill(past, 0)
    f, count = _ragged(m.backbone.layers, z[:, None], u)
    keep = torch.arange(f.shape[-1], device=device) < count[:, None]
    pooled = torch.where(keep[:, None], f, 0.).sum(-1)/count.clamp(min=1)[:, None].float()
    out = m.logits(pooled, u).masked_fill((u < m.min_samples)[:, None], float('nan'))
    return out.cpu()


def incremental(bank, slots, starts, chunks):
    bank.step(slots, starts, chunks)
    return bank.readout(slots).cpu()


def timed(fn, device):
    sync(device); t0 = time.perf_counter()
    out = fn()
    sync(device)
    return time.perf_counter() - t0, out


def run(a):
    global CHUNKS, BATCHES, REPS
    if a.smoke: CHUNKS, BATCHES, REPS = (1600, 18000), (1, 8), 1
    device = torch.device(a.device)
    torch.set_num_threads(4)
    if device.type == 'cuda': torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    model = load_stream_r0(a.model, device)[0].eval()
    cap = model.max_samples
    data = load_prefix_cache(Path(a.data)/'test', verify=False)
    long_ = np.flatnonzero(np.asarray(data.raw_length) >= cap)
    rng = np.random.default_rng(SEED)
    pool = rng.permutation(long_)
    need = max(BATCHES)*(REPS + 1)
    if len(pool) < need: raise ValueError(f'{len(pool)} reads >= {cap}, need {need}')
    pa_all = np.asarray(data.pa(np.sort(pool[:need]), 0, cap), np.float32)    # [need, cap] pA
    a.out.mkdir(parents=True, exist_ok=True)
    meta = dict(model=str(a.model), data=str(a.data), device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu', cap=cap, skip=model.skip,
                min_decision_raw=model.skip + model.min_samples, reads_available=int(len(long_)), torch=torch.__version__,
                chunks=CHUNKS, batches=BATCHES, reps=REPS, seed=SEED, started=time.strftime('%Y-%m-%d %H:%M:%S'))
    rows, worst = [], 0.
    for chunk in CHUNKS:
        for batch in BATCHES:
            for rep in range(REPS + 1):                     # rep 0: warm-up, not recorded
                pa = pa_all[rep*batch:(rep + 1)*batch]
                # aligned: every read gets the same chunks
                bank = StreamR0Bank(model, batch, device); slots = np.array([bank.allocate() for _ in range(batch)])
                n0 = 0
                while n0 < cap:
                    n1 = min(n0 + chunk, cap)
                    t_inc, lo_inc = timed(lambda: incremental(bank, slots, [n0]*batch, list(pa[:, n0:n1])), device)
                    n = np.full(batch, n1)
                    if n1 >= model.skip + model.min_samples:
                        t_scr, lo_scr = timed(lambda: scratch(model, pa, n, device), device)
                        ok = torch.isfinite(lo_inc)
                        diff = float((lo_inc[ok] - lo_scr[ok]).abs().max()) if ok.any() else 0.
                        if rep: worst = max(worst, diff)
                        if rep:
                            rows.append(dict(scenario='aligned', chunk=chunk, batch=batch, rep=rep, n=int(n1), method='incremental', seconds=t_inc))
                            rows.append(dict(scenario='aligned', chunk=chunk, batch=batch, rep=rep, n=int(n1), method='scratch', seconds=t_scr, max_abs_logit_diff=diff))
                    n0 = n1
                # mixed: random positions, one timed chunk update
                n = rng.integers(max(chunk, model.skip + model.min_samples), cap + 1, batch)
                bank = StreamR0Bank(model, batch, device); slots = np.array([bank.allocate() for _ in range(batch)])
                if (n > chunk).any(): bank.step(slots, [0]*batch, [pa[i, :n[i] - chunk] if n[i] > chunk else None for i in range(batch)])
                sync(device)
                t_inc, lo_inc = timed(lambda: incremental(bank, slots, list(n - chunk), [pa[i, n[i] - chunk:n[i]] for i in range(batch)]), device)
                t_scr, lo_scr = timed(lambda: scratch(model, pa, n, device), device)
                diff = float((lo_inc - lo_scr).abs().nan_to_num(0).max())
                if rep:
                    worst = max(worst, diff)
                    rows.append(dict(scenario='mixed', chunk=chunk, batch=batch, rep=rep, n=int(np.median(n)), method='incremental', seconds=t_inc))
                    rows.append(dict(scenario='mixed', chunk=chunk, batch=batch, rep=rep, n=int(np.median(n)), method='scratch', seconds=t_scr, max_abs_logit_diff=diff))
                del bank
            peak = torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.
            print(time.strftime('%H:%M:%S'), 'chunk', chunk, 'batch', batch, 'worst logit diff', f'{worst:.2e}', 'peak GiB', round(peak, 2), flush=True)
    meta.update(finished=time.strftime('%Y-%m-%d %H:%M:%S'), max_abs_logit_diff=worst,
                peak_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.)
    (a.out/'timing.json').write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))


def report(a):
    d = json.loads((a.out/'timing.json').read_text()); meta, rows = d['meta'], d['rows']
    chunks, batches, reps = tuple(meta['chunks']), tuple(meta['batches']), meta['reps']
    t = {}
    for r in rows:   # aligned: keyed by n; mixed: one key per (chunk, batch, method), n differs by repetition
        key = (r['scenario'], r['chunk'], r['batch'], r['n'] if r['scenario'] == 'aligned' else None, r['method'])
        t.setdefault(key, []).append(r['seconds'])
    ms = lambda *k: float(np.median(t[k]))*1000 if k in t else float('nan')
    cell = lambda i_, s_: f'{i_:.1f} / {s_:.1f}（{s_/i_:.1f}×）'
    full = lambda c: max(k[3] for k in t if k[:2] == ('aligned', c) and k[3] % c == 0)   # last update adding a whole chunk
    L = [f"# 增量计算 vs 从头计算：每次 chunk 更新的耗时（r10_run01，{meta['gpu']}）", '',
         f"模型：预算 18000 的 S1 编码器（StreamR0，上限 {meta['cap']} 点，4 kHz）。数据：r10_run01 TEST 中长度 ≥ {meta['cap']} 的 read，"
         f"每个 (chunk, batch, 重复) 用互不重复的 read。每格 {reps} 次重复取中位数，另有 1 次预热不计。",
         '一次更新 = B 条 read 各收到一个 chunk 后得到 logit：增量 = StreamR0Bank.step + readout；从头 = 把已收到的整段前缀上传，'
         '因果归一化后用原始主干整段前向、全帧平均、过 head。两者都含 CPU→GPU 上传和 logit 取回 CPU，独占一张 GPU。',
         f"正确性：每一步核对两种算法的 logit，最大绝对差 {meta['max_abs_logit_diff']:.1e}。显存峰值 {meta['peak_gib']:.1f} GiB。", '',
         '## 汇总：同步到达时，最后一个完整 chunk 的更新耗时（已收到信号最长、且这一步新增整整一个 chunk），增量 / 从头 ms（倍数）', '',
         '| chunk | 已收到 | ' + ' | '.join(f'B={b}' for b in batches) + ' |', '|---:|---:|' + '---:|'*len(batches)]
    for c in chunks:
        n = full(c)
        L.append(f'| {c} 点（{c/FS:g} s） | {n} 点 | ' + ' | '.join(cell(ms('aligned', c, b, n, 'incremental'), ms('aligned', c, b, n, 'scratch'))
                                                                  for b in batches) + ' |')
    L += ['', '## 稳态（每条 read 处在 U[max(chunk, 1932), 19500] 的随机位置），增量 / 从头 ms（倍数）', '',
          '| chunk | ' + ' | '.join(f'B={b}' for b in batches) + ' |', '|---:|' + '---:|'*len(batches)]
    for c in chunks:
        L.append(f'| {c} 点（{c/FS:g} s） | ' + ' | '.join(cell(ms('mixed', c, b, None, 'incremental'), ms('mixed', c, b, None, 'scratch'))
                                                          for b in batches) + ' |')
    L += ['', '## 稳态下一次更新占 chunk 间隔的比例（< 100% 才跟得上实时），增量 / 从头', '',
          '| chunk | ' + ' | '.join(f'B={b}' for b in batches) + ' |', '|---:|' + '---:|'*len(batches)]
    for c in chunks:
        dt = c/FS*1000
        L.append(f'| {c} 点（{c/FS:g} s） | ' + ' | '.join(
            f"{ms('mixed', c, b, None, 'incremental')/dt*100:.1f}% / {ms('mixed', c, b, None, 'scratch')/dt*100:.1f}%" for b in batches) + ' |')
    L += ['', '## 同步到达：随已收到信号长度的变化（所有 read 同时收到 chunk），增量 / 从头 ms（倍数）', '']
    for c in chunks:
        L += [f'### chunk {c} 点（{c/FS:g} s）', '', '| 已收到 | ' + ' | '.join(f'B={b}' for b in batches) + ' |',
              '|---:|' + '---:|'*len(batches)]
        for n in sorted({k[3] for k in t if k[:2] == ('aligned', c)}):
            L.append(f'| {n} 点（{n/FS:.2f} s） | ' + ' | '.join(cell(ms('aligned', c, b, n, 'incremental'), ms('aligned', c, b, n, 'scratch'))
                                                              for b in batches) + ' |')
        L.append('')
    (a.out/'summary_zh.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run'); p.add_argument('--data', required=True); p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--smoke', action='store_true', help='chunks 1600 / 18000, batches 1 / 8, one repetition')
    p = sub.add_parser('report'); p.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); globals()[a.cmd](a)


if __name__ == '__main__':
    main()
