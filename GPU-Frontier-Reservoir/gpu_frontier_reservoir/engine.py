from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch


@dataclass
class SearchConfig:
    n: int
    r: int
    p: int
    frontier_length: int
    target_length: int
    bucket_size: int
    use_best: int
    save_best: int
    degree_window: int
    shard_count: int = 1
    shard_index: int = 0
    seed: int = 1
    expansion_chunk: int = 50_000
    matmul_chunk: int = 4_000
    boundary_margin: int = 16
    device: str = "cuda"

    @property
    def dim(self) -> int:
        return math.comb(self.n, self.r) - math.comb(self.n, self.r - 1)

    def validate(self) -> None:
        if self.n < 2 or self.r < 1 or self.n < 2 * self.r:
            raise ValueError("require n >= 2r and r >= 1")
        if self.p <= 1:
            raise ValueError("GPU modular search requires p > 1")
        if not 1 <= self.frontier_length <= self.target_length:
            raise ValueError("require 1 <= frontier_length <= target_length")
        if min(self.bucket_size, self.use_best, self.save_best, self.degree_window) <= 0:
            raise ValueError("bucket/use_best/save_best/degree_window must be positive")
        if not 0 <= self.shard_index < self.shard_count:
            raise ValueError("invalid shard index")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_tables(path: Path, cfg: SearchConfig):
    tables = torch.load(path, map_location="cpu", weights_only=True)
    for key, expected in (("n", cfg.n), ("r", cfg.r), ("p", cfg.p), ("dim", cfg.dim)):
        if int(tables[key]) != expected:
            raise ValueError(f"table {key}={tables[key]} but search requires {expected}")
    raw = tables["simple_burau"]
    center = int(tables["center"])
    simple = torch.zeros(raw.shape[0], cfg.dim, cfg.dim, cfg.degree_window, dtype=torch.int16)
    for s in range(raw.shape[0]):
        mask = raw[s].abs().sum((0, 1)) != 0
        if not mask.any():
            continue
        lo, hi = torch.where(mask)[0][[0, -1]].tolist()
        min_degree = lo - center
        width = hi - lo + 1
        if min_degree < 0 or min_degree + width > cfg.degree_window:
            raise ValueError(f"simple {s} does not fit nonnegative search window")
        simple[s, ..., min_degree:min_degree + width] = raw[s, ..., lo:hi + 1].to(torch.int16)
    valid = tables["valid_suffixes"].to(torch.int32).clone()
    counts = tables["num_valid_suffixes"].to(torch.int32).clone()
    # Match beta/paper initialization: the empty word has the legal followers of Delta.
    delta = int(tables["delta_index"])
    identity = int(tables["id_index"])
    valid[identity] = valid[delta]
    counts[identity] = counts[delta]
    return simple, valid, counts


def iter_frontier(valid: torch.Tensor, counts: torch.Tensor, depth: int) -> Iterator[tuple[int, ...]]:
    """Enumerate the same positive GNF frontier represented by the beta suffix table."""
    # load_tables patches identity (index zero in the supported tables) to the
    # legal first factors, exactly as the beta loader does.
    first = valid[0, : int(counts[0])].tolist()
    stack = [(tuple(), iter(int(x) for x in first))]
    while stack:
        prefix, choices = stack[-1]
        try:
            nxt = next(choices)
        except StopIteration:
            stack.pop()
            continue
        word = prefix + (nxt,)
        if len(word) == depth:
            yield word
        else:
            children = valid[nxt, : int(counts[nxt])].tolist()
            stack.append((word, iter(int(x) for x in children)))


def projectivize_batch(mats: torch.Tensor, D: int):
    """Shift each nonzero polynomial matrix left by its valuation, like peyl.projectivise."""
    support = mats.ne(0).any(dim=(1, 2))
    nonzero = support.any(dim=1)
    starts = support.to(torch.int32).argmax(dim=1)
    ends = D - support.flip(1).to(torch.int32).argmax(dim=1)
    projlens = torch.where(nonzero, ends - starts, torch.zeros_like(starts))
    idx = torch.arange(D, device=mats.device)[None, :] + starts[:, None]
    valid = idx < D
    idx = idx.clamp(max=D - 1)
    out = mats.gather(3, idx[:, None, None, :].expand(-1, mats.shape[1], mats.shape[2], -1))
    out *= valid[:, None, None, :]
    return out, projlens, ends


def scalar_identity_mask(mats: torch.Tensor) -> torch.Tensor:
    dim = mats.shape[1]
    diag = mats[:, 0, 0]
    ok = torch.ones(len(mats), dtype=torch.bool, device=mats.device)
    for i in range(dim):
        ok &= mats[:, i, i].eq(diag).all(1)
        for j in range(dim):
            if i != j:
                ok &= mats[:, i, j].eq(0).all(1)
    return ok & diag.ne(0).any(1)


class FFTMultiplier:
    def __init__(self, simples: torch.Tensor, device: torch.device):
        self.device = device
        self.D = simples.shape[-1]
        self.dim = simples.shape[1]
        self.fft_size = 1 << (2 * self.D - 2).bit_length()
        self.simple_fft = torch.fft.rfft(simples.float().to(device), n=self.fft_size, dim=-1)

    def multiply(self, parents: torch.Tensor, suffixes: torch.Tensor, p: int, chunk: int):
        # Do not accumulate int32 chunk results in a Python list and concatenate:
        # for wide windows that temporarily doubles peak memory.  Coefficients
        # are reduced modulo p, so compact int16 is sufficient for storage.
        output = torch.empty(
            len(parents), self.dim, self.dim, self.D,
            dtype=torch.int16, device=self.device,
        )
        for start in range(0, len(parents), chunk):
            end = min(start + chunk, len(parents))
            A = torch.fft.rfft(parents[start:end].float(), n=self.fft_size, dim=-1)
            B = self.simple_fft[suffixes[start:end].long()]
            C = torch.einsum("nikf,nkjf->nijf", A, B)
            real = torch.fft.irfft(C, n=self.fft_size, dim=-1)[..., :self.D]
            output[start:end] = torch.round(real).to(torch.int32).remainder_(p).to(torch.int16)
            del A, B, C, real
        return output


class Reservoir:
    """One uniform priority reservoir per projlen bucket."""
    def __init__(self, capacity: int, generator: torch.Generator):
        self.capacity, self.generator = capacity, generator
        self.data: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.seen: dict[int, int] = {}

    def add(self, mats, words, projlens):
        # Reservoir state belongs in CPU RAM. Keeping both the selected parent
        # generation and growing child generation on a 44GB GPU is impossible
        # for wide 9x9 polynomial matrices and large paper-style buckets.
        mats = mats.to(device="cpu", dtype=torch.int16)
        words = words.to(device="cpu", dtype=torch.int32)
        projlens = projlens.to(device="cpu")
        priorities = torch.rand(len(mats), generator=self.generator)
        for pl in torch.unique(projlens).tolist():
            mask = projlens == pl
            nm, nw, np = mats[mask].to(torch.int16), words[mask].to(torch.int32), priorities[mask]
            self.seen[pl] = self.seen.get(pl, 0) + len(nm)
            if pl in self.data:
                om, ow, op = self.data[pl]
                nm, nw, np = torch.cat((om, nm)), torch.cat((ow, nw)), torch.cat((op, np))
            if len(nm) > self.capacity:
                keep = torch.topk(np, self.capacity).indices
                nm, nw, np = nm[keep], nw[keep], np[keep]
            self.data[int(pl)] = nm, nw, np

    def select(self, limit: int):
        mats, words, used = [], [], 0
        for pl in sorted(self.data):
            m, w, _ = self.data[pl]
            if used + len(m) > limit:
                break  # paper behavior: select complete buckets only
            mats.append(m); words.append(w); used += len(m)
        if not mats:
            raise RuntimeError("use_best is smaller than the lowest complete bucket")
        # Reservoir matrices are already reduced modulo p. Keep int16 here;
        # promoting a large use_best selection to int32 can consume tens of GB.
        return torch.cat(mats), torch.cat(words), used


class Search:
    def __init__(self, cfg: SearchConfig, table_path: Path, output: Path):
        cfg.validate()
        self.cfg, self.table_path, self.output = cfg, table_path, output
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
        self.rng = torch.Generator(device="cpu").manual_seed(cfg.seed)
        simple, valid, counts = load_tables(table_path, cfg)
        self.simple = simple.to(self.device)
        self.valid, self.counts = valid.to(self.device), counts.to(self.device)
        self.valid_cpu, self.counts_cpu = valid, counts
        self.multiplier = FFTMultiplier(simple, self.device)
        self.db = sqlite3.connect(output / "good_braids.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS good_braids(length INT, projlen INT, factors TEXT, matrix_sha256 TEXT)")
        (output / "config.json").write_text(json.dumps({**asdict(cfg), "table": str(table_path), "table_sha256": sha256_file(table_path)}, indent=2, sort_keys=True))

    def _evaluate_words(self, words: torch.Tensor):
        mats = torch.zeros(len(words), self.cfg.dim, self.cfg.dim, self.cfg.degree_window, dtype=torch.int32, device=self.device)
        eye = torch.arange(self.cfg.dim, device=self.device)
        mats[:, eye, eye, 0] = 1
        for pos in range(words.shape[1]):
            mats = self.multiplier.multiply(mats, words[:, pos], self.cfg.p, self.cfg.matmul_chunk)
            mats, _, ends = projectivize_batch(mats, self.cfg.degree_window)
            if (ends >= self.cfg.degree_window).any():
                raise RuntimeError("degree window overflow while evaluating frontier")
        return mats

    def build_frontier(self):
        reservoir = Reservoir(self.cfg.bucket_size, self.rng)
        batch = []
        total = assigned = 0
        for word in iter_frontier(self.valid_cpu, self.counts_cpu, self.cfg.frontier_length):
            if total % self.cfg.shard_count == self.cfg.shard_index:
                batch.append(word); assigned += 1
            total += 1
            if len(batch) >= self.cfg.expansion_chunk:
                self._add_frontier_batch(reservoir, batch); batch = []
        if batch: self._add_frontier_batch(reservoir, batch)
        return reservoir, total, assigned

    def _add_frontier_batch(self, reservoir, batch):
        words = torch.tensor(batch, dtype=torch.int32, device=self.device)
        mats = self._evaluate_words(words)
        mats, pls, _ = projectivize_batch(mats, self.cfg.degree_window)
        reservoir.add(mats, words, pls)

    def _save_best(self, length: int, reservoir: Reservoir):
        saved = 0
        for pl in sorted(reservoir.data):
            mats, words, _ = reservoir.data[pl]
            if saved + len(words) > self.cfg.save_best: break
            cpu_w, cpu_m = words.cpu(), mats.cpu()
            rows = []
            for word, mat in zip(cpu_w, cpu_m):
                h = hashlib.sha256(mat.numpy().tobytes()).hexdigest()
                rows.append((length, pl, json.dumps(word.tolist()), h))
            self.db.executemany("INSERT INTO good_braids VALUES (?,?,?,?)", rows)
            saved += len(rows)
        self.db.commit()
        return saved

    def _expand(self, current: Reservoir, length: int):
        # Selection is assembled in CPU RAM. Only each expansion chunk is sent
        # to CUDA, leaving device memory for FFT workspaces and results.
        parents, words, selected = current.select(self.cfg.use_best)
        current.data.clear()
        last = words[:, -1].long()
        counts = self.counts_cpu[last].long()
        parent_idx = torch.repeat_interleave(torch.arange(len(words)), counts)
        ends = torch.cumsum(counts, 0); starts = ends - counts
        local = torch.arange(int(ends[-1])) - starts[parent_idx]
        suffix = self.valid_cpu[last[parent_idx], local].long()
        nxt = Reservoir(self.cfg.bucket_size, self.rng)
        boundary = kernels = candidates = 0
        for start in range(0, len(parent_idx), self.cfg.expansion_chunk):
            idx = parent_idx[start:start+self.cfg.expansion_chunk]
            suf_cpu = suffix[start:start+self.cfg.expansion_chunk]
            parent_gpu = parents[idx].to(self.device, non_blocking=True)
            suf = suf_cpu.to(self.device, non_blocking=True)
            mats = self.multiplier.multiply(parent_gpu, suf, self.cfg.p, self.cfg.matmul_chunk)
            mats, pls, raw_ends = projectivize_batch(mats, self.cfg.degree_window)
            boundary += int((pls >= self.cfg.degree_window - self.cfg.boundary_margin).sum())
            kmask = scalar_identity_mask(mats)
            kernels += int(kmask.sum())
            child_words = torch.cat((words[idx], suf_cpu[:, None].to(torch.int32)), 1)
            if kmask.any():
                with (self.output / "kernel_candidates.jsonl").open("a") as f:
                    for w in child_words[kmask.cpu()].tolist():
                        f.write(json.dumps({"length": length, "factors": w, "gpu_scalar_identity": True}) + "\n")
            nxt.add(mats, child_words, pls); candidates += len(mats)
            del parent_gpu, suf, mats, pls
        if boundary:
            raise RuntimeError(f"{boundary} candidates entered degree-window safety margin; increase --degree-window")
        return nxt, selected, candidates, kernels

    def run(self):
        start = time.time()
        status = "clean"
        try:
            current, total, assigned = self.build_frontier()
            with (self.output / "progress.jsonl").open("a") as progress:
                for length in range(self.cfg.frontier_length, self.cfg.target_length + 1):
                    saved = self._save_best(length, current)
                    best = min(current.data)
                    row = {"length": length, "best_projlen": best, "bucket_count": len(current.data), "live": sum(len(x[0]) for x in current.data.values()), "saved": saved, "frontier_total": total, "frontier_assigned": assigned, "elapsed_seconds": time.time()-start}
                    print(json.dumps(row, sort_keys=True), flush=True); progress.write(json.dumps(row, sort_keys=True)+"\n"); progress.flush()
                    if length == self.cfg.target_length: break
                    current, selected, candidates, kernels = self._expand(current, length + 1)
                    row.update({"selected": selected, "candidates": candidates, "kernel_candidates": kernels})
        except Exception:
            status = "malformed"
            raise
        finally:
            self.db.close()
            (self.output / "status.json").write_text(json.dumps({"status": status, "elapsed_seconds": time.time()-start}, indent=2))
