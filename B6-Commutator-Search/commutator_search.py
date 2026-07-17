#!/usr/bin/env python3
"""GPU paper-reservoir search for [sigma_i, g^-1] in a two-row Jones representation.

This is a dimension-independent port of the successful B4 commutator search.
It uses the projective update C(gb) = T_b C(g) M_b, where
T_b = M_sigma M_b^-1 M_sigma^-1.  The M_b matrices come from a beta table;
the T_b matrices are evaluated exactly with peyl and cached on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GPU_PROJECT = REPO / "GPU-Frontier-Reservoir"
# This vendored copy treats pandas-dependent braidsearch as optional.  That is
# important on the CUDA-13 environment, whose pandas installation is not
# usable; the commutator engine only needs braid, jonesrep, and polymat.
PEYL_ROOT = (
    REPO / "Tried_algorithms" / "structural-kernel-experiments"
    / "third_party" / "braids_project"
)
for path in (GPU_PROJECT, PEYL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gpu_frontier_reservoir.engine import (  # noqa: E402
    Reservoir,
    Search,
    SearchConfig,
    projectivize_batch,
    scalar_identity_mask,
    sha256_file,
)
from peyl.braid import GNF, PermTable  # noqa: E402
from peyl.jonesrep import JonesCellRep  # noqa: E402
from peyl import polymat  # noqa: E402


def index_to_perm(index: int, n: int) -> tuple[int, ...]:
    available = list(range(n))
    result = []
    for i in range(n):
        q, index = divmod(index, math.factorial(n - 1 - i))
        result.append(available.pop(q))
    return tuple(result)


def perm_to_index(perm: tuple[int, ...]) -> int:
    available = list(range(len(perm)))
    result = 0
    for i, value in enumerate(perm):
        pos = available.index(value)
        result += pos * math.factorial(len(perm) - 1 - i)
        available.pop(pos)
    return result


def sigma_factor(n: int, generator: int) -> int:
    perm = list(range(n))
    perm[generator - 1], perm[generator] = perm[generator], perm[generator - 1]
    return perm_to_index(tuple(perm))


def left_descent_set(perm: tuple[int, ...]) -> set[int]:
    inverse = [0] * len(perm)
    for i, value in enumerate(perm):
        inverse[value] = i
    return {i for i in range(len(perm) - 1) if inverse[i] > inverse[i + 1]}


def allowed_first_factors(n: int, generator: int, identity: int, delta: int) -> list[int]:
    s = generator - 1
    # Same sufficient centralizer-coset avoidance condition as the B4 code.
    parabolic = {j for j in range(n - 1) if j == s or abs(j - s) >= 2}
    return [
        i for i in range(math.factorial(n))
        if i not in (identity, delta)
        and left_descent_set(index_to_perm(i, n)).isdisjoint(parabolic)
    ]


def compact_pack(matrices: list[np.ndarray], p: int) -> torch.Tensor:
    matrices = [polymat.projectivise(m % p) for m in matrices]
    width = max(matrix.shape[-1] for matrix in matrices)
    dim = matrices[0].shape[0]
    packed = torch.zeros(len(matrices), dim, dim, width, dtype=torch.int16)
    for i, matrix in enumerate(matrices):
        packed[i, ..., : matrix.shape[-1]] = torch.from_numpy(matrix.astype(np.int16))
    return packed


def build_twisted_cache(table_path: Path, cache_path: Path, generator: int) -> dict:
    table = torch.load(table_path, map_location="cpu", weights_only=True)
    n, r, p = (int(table[key]) for key in ("n", "r", "p"))
    identity = int(table["id_index"])
    delta = int(table["delta_index"])
    if not 1 <= generator < n:
        raise ValueError(f"generator must be in 1..{n-1}")
    rep = JonesCellRep(n=n, r=r, p=p)
    sf = sigma_factor(n, generator)
    sigma = GNF(n=n, power=0, factors=(sf,))
    twisted = []
    started = time.time()
    for factor in range(math.factorial(n)):
        # Identity and Delta are distinguished entries in the permutation
        # table, but neither is a legal canonical factor in a GNF factors
        # tuple.  Represent them by their canonical GNF forms instead.
        if factor == identity:
            b = GNF.identity(n)
        elif factor == delta:
            b = GNF(n=n, power=1, factors=())
        else:
            b = GNF(n=n, power=0, factors=(factor,))
        braid = sigma * b.inv() * sigma.inv()
        twisted.append(rep.polymat_evaluate_braid(braid))
        if factor and factor % 100 == 0:
            print(f"  built {factor}/{math.factorial(n)} twisted matrices", flush=True)
    payload = {
        "format": "two-row-commutator-twisted-v1",
        "n": n, "r": r, "p": p, "dim": int(table["dim"]),
        "generator": generator, "sigma_factor": sf,
        "source_table": str(table_path),
        "source_table_sha256": sha256_file(table_path),
        "twisted": compact_pack(twisted, p),
        "elapsed_seconds": time.time() - started,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"Wrote {cache_path} ({sha256_file(cache_path)})", flush=True)
    return payload


class TwoSidedFFT:
    def __init__(self, simple: torch.Tensor, twisted: torch.Tensor, device: torch.device, max_D: int):
        self.simple = simple.to(device)
        self.twisted = twisted.to(device)
        self.device, self.max_D = device, max_D
        self.dim = simple.shape[1]
        # Compatibility with the shared reservoir banner. Here this is the
        # compact support width of the positive-simple table.
        self.simple_D = simple.shape[-1]
        self.cache: dict[tuple[str, int], torch.Tensor] = {}

    @property
    def cache_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.cache.values())

    def _fft(self, which: str, size: int) -> torch.Tensor:
        key = (which, size)
        if key not in self.cache:
            base = self.simple if which == "simple" else self.twisted
            self.cache[key] = torch.fft.rfft(base.float(), n=size, dim=-1)
        return self.cache[key]

    def _multiply(self, parents, suffixes, p: int, chunk: int, *, left: bool):
        base = self.twisted if left else self.simple
        natural_D = parents.shape[-1] + base.shape[-1] - 1
        if natural_D > self.max_D:
            raise RuntimeError(
                f"degree window overflow: product needs {natural_D}, cap is {self.max_D}"
            )
        fft_size = 1 << (natural_D - 1).bit_length()
        fixed = self._fft("twisted" if left else "simple", fft_size)
        output = torch.empty(
            len(parents), self.dim, self.dim, natural_D,
            dtype=torch.int16, device=self.device,
        )
        for start in range(0, len(parents), chunk):
            end = min(start + chunk, len(parents))
            moving = torch.fft.rfft(parents[start:end].float(), n=fft_size, dim=-1)
            chosen = fixed[suffixes[start:end].long()]
            product = (
                torch.einsum("nikf,nkjf->nijf", chosen, moving)
                if left else torch.einsum("nikf,nkjf->nijf", moving, chosen)
            )
            real = torch.fft.irfft(product, n=fft_size, dim=-1)[..., :natural_D]
            output[start:end] = torch.round(real).to(torch.int32).remainder_(p).to(torch.int16)
            del moving, chosen, product, real
        return output

    def update(self, parents, suffixes, p: int, chunk: int):
        first = self._multiply(parents, suffixes, p, chunk, left=True)
        first, first_projlens, _ = projectivize_batch(first, first.shape[-1])
        # projectivize_batch shifts support but deliberately preserves its input
        # allocation.  Trim the all-zero tail before the second product so the
        # FFT size follows actual projlen rather than accumulated workspace size.
        first = first[..., : max(1, int(first_projlens.max().item()))]
        second = self._multiply(first, suffixes, p, chunk, left=False)
        del first
        second, projlens, ends = projectivize_batch(second, second.shape[-1])
        return second, projlens, ends


class CommutatorSearch(Search):
    def __init__(self, cfg, table_path, twisted_path, output, generator):
        self.generator = generator
        super().__init__(cfg, table_path, output)
        cache = torch.load(twisted_path, map_location="cpu", weights_only=True)
        expected = {"n": cfg.n, "r": cfg.r, "p": cfg.p, "dim": cfg.dim, "generator": generator}
        for key, value in expected.items():
            if int(cache[key]) != value:
                raise ValueError(f"twisted cache {key}={cache[key]}, expected {value}")
        if cache["source_table_sha256"] != sha256_file(table_path):
            raise ValueError("twisted cache was made from a different table artifact")
        self.twisted_path = twisted_path
        self.multiplier = TwoSidedFFT(self.simple.cpu(), cache["twisted"], self.device, cfg.degree_window)
        raw = torch.load(table_path, map_location="cpu", weights_only=True)
        self.identity, self.delta = int(raw["id_index"]), int(raw["delta_index"])
        self.first = allowed_first_factors(cfg.n, generator, self.identity, self.delta)
        self.rep = JonesCellRep(n=cfg.n, r=cfg.r, p=cfg.p)
        self.sigma = GNF(n=cfg.n, power=0, factors=(sigma_factor(cfg.n, generator),))
        config_path = self.output / "config.json"
        metadata = json.loads(config_path.read_text())
        metadata.update({
            "method": "commutator_paper_reservoir",
            "generator": generator,
            "twisted_cache": str(twisted_path),
            "twisted_cache_sha256": sha256_file(twisted_path),
        })
        config_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    def _print_banner(self):
        super()._print_banner()
        print(f"Objective: projlen of [sigma_{self.generator}, g^-1]")
        print(f"Allowed first factors: {len(self.first)}")
        print(f"Twisted cache: {self.twisted_path}")
        print("Update: C(gb) = T_b C(g) M_b (projectivized after each product)", flush=True)

    def _iter_frontier(self):
        stack = [(tuple(), iter(self.first))]
        while stack:
            prefix, choices = stack[-1]
            try:
                nxt = next(choices)
            except StopIteration:
                stack.pop(); continue
            word = prefix + (int(nxt),)
            if len(word) == self.cfg.frontier_length:
                yield word
            else:
                count = int(self.counts_cpu[nxt])
                stack.append((word, iter(self.valid_cpu[nxt, :count].tolist())))

    def _evaluate_words(self, words: torch.Tensor):
        mats = torch.zeros(len(words), self.cfg.dim, self.cfg.dim, 1, dtype=torch.int16, device=self.device)
        diagonal = torch.arange(self.cfg.dim, device=self.device)
        mats[:, diagonal, diagonal, 0] = 1
        for pos in range(words.shape[1]):
            mats, _, _ = self.multiplier.update(mats, words[:, pos], self.cfg.p, self.cfg.matmul_chunk)
        return mats

    def _add_frontier_batch(self, reservoir, batch):
        words = torch.tensor(batch, dtype=torch.int32, device=self.device)
        mats = self._evaluate_words(words)
        mats, projlens, _ = projectivize_batch(mats, mats.shape[-1])
        kernel_mask = scalar_identity_mask(mats)
        if kernel_mask.any():
            self._record_candidates(words[kernel_mask].cpu(), words.shape[1])
            # This matches the original B4 commutator engine: every scalar
            # identity is a terminal state. Genuine kernels have already been
            # saved, while trivial commutators must not monopolize projlen=1.
            keep = ~kernel_mask
            mats, words, projlens = mats[keep], words[keep], projlens[keep]
        if len(mats):
            reservoir.add(mats, words, projlens)

    def build_frontier(self):
        started = time.time()
        reservoir = Reservoir(self.cfg.bucket_size, self.rng)
        batch, total, assigned = [], 0, 0
        for word in self._iter_frontier():
            if total % self.cfg.shard_count == self.cfg.shard_index:
                batch.append(word); assigned += 1
            total += 1
            if len(batch) >= self.cfg.expansion_chunk:
                self._add_frontier_batch(reservoir, batch); batch = []
        if batch:
            self._add_frontier_batch(reservoir, batch)
        return reservoir, total, assigned, time.time() - started

    def _verify(self, factors: list[int]) -> dict:
        g = GNF(n=self.cfg.n, power=0, factors=tuple(factors))
        comm = self.sigma * g.inv() * self.sigma.inv() * g
        if comm == GNF.identity(self.cfg.n):
            return {"verified": False, "reason": "trivial_commutator"}
        matrix = self.rep.polymat_evaluate_braid(comm) % self.cfg.p
        diagonal = matrix[0, 0]
        scalar = all(
            np.array_equal(matrix[i, j], diagonal) if i == j else not np.any(matrix[i, j])
            for i in range(self.cfg.dim) for j in range(self.cfg.dim)
        )
        return {
            "verified": bool(scalar),
            "reason": "exact_scalar_identity" if scalar else "gpu_false_positive",
            "commutator_factors": list(comm.factors),
            "commutator_power": int(comm.power),
            "matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        }

    def _record_candidates(self, words: torch.Tensor, length: int):
        counts = {"gpu_candidates": len(words), "verified": 0,
                  "trivial_commutators": 0, "gpu_false_positives": 0}
        verified_path = self.output / "verified_kernels.jsonl"
        with verified_path.open("a") as verified_handle:
            for factors in words.tolist():
                result = self._verify(factors)
                if result["verified"]:
                    counts["verified"] += 1
                    record = {
                        "n": self.cfg.n, "r": self.cfg.r, "p": self.cfg.p,
                        "representation": [self.cfg.n-self.cfg.r, self.cfg.r],
                        "generator": self.generator, "length": length,
                        "factors": factors, "projlen": 1, **result,
                    }
                    verified_handle.write(json.dumps(record, sort_keys=True) + "\n")
                elif result["reason"] == "trivial_commutator":
                    counts["trivial_commutators"] += 1
                else:
                    counts["gpu_false_positives"] += 1
        with (self.output / "candidate_summary.jsonl").open("a") as handle:
            handle.write(json.dumps({"length": length, **counts}, sort_keys=True) + "\n")
        print(
            f"  Scalar candidates: {counts['gpu_candidates']}; "
            f"verified kernels: {counts['verified']}; "
            f"trivial: {counts['trivial_commutators']}; "
            f"GPU false positives: {counts['gpu_false_positives']}",
            flush=True,
        )
        return counts

    def _expand(self, current: Reservoir, length: int):
        level_start = time.time()
        parents, words, selected = current.select(self.cfg.use_best)
        current.data.clear()
        last = words[:, -1].long()
        counts = self.counts_cpu[last].long()
        parent_idx = torch.repeat_interleave(torch.arange(len(words)), counts)
        ends = torch.cumsum(counts, 0); starts = ends - counts
        local = torch.arange(int(ends[-1])) - starts[parent_idx]
        suffix = self.valid_cpu[last[parent_idx], local].long()
        print("=" * 60)
        print(f"Level {length} - COMMUTATOR SAMPLING")
        print("=" * 60)
        print(f"  Starting braids: {selected}")
        print(f"  Candidates to generate: {len(parent_idx)}")
        chunks = math.ceil(len(parent_idx) / self.cfg.expansion_chunk)
        if chunks > 1:
            print(f"  Processing in {chunks} chunks...", flush=True)
        nxt = Reservoir(self.cfg.bucket_size, self.rng)
        candidates = kernels = 0
        matmul_seconds = sampling_seconds = 0.0
        for start in range(0, len(parent_idx), self.cfg.expansion_chunk):
            idx = parent_idx[start:start+self.cfg.expansion_chunk]
            suf_cpu = suffix[start:start+self.cfg.expansion_chunk]
            parent_gpu = parents[idx].to(self.device, non_blocking=True)
            suf = suf_cpu.to(self.device, non_blocking=True)
            tick = time.time()
            mats, pls, _ = self.multiplier.update(parent_gpu, suf, self.cfg.p, self.cfg.matmul_chunk)
            matmul_seconds += time.time() - tick
            child_words = torch.cat((words[idx], suf_cpu[:, None].to(torch.int32)), 1)
            kmask = scalar_identity_mask(mats)
            if kmask.any():
                hit_words = child_words[kmask.cpu()]
                kernels += len(hit_words)
                self._record_candidates(hit_words, length)
                # Scalar identities are terminal, exactly as in the original
                # B4 search. Do not let trivial centralizer states refill the
                # lowest reservoir bucket at every subsequent level.
                keep = ~kmask
                mats, pls = mats[keep], pls[keep]
                child_words = child_words[keep.cpu()]
            tick = time.time()
            if len(mats):
                nxt.add(mats, child_words, pls)
            sampling_seconds += time.time() - tick
            candidates += len(mats)
            del parent_gpu, suf, mats, pls
        self._print_level_result(
            nxt, matmul_seconds=matmul_seconds, sampling_seconds=sampling_seconds,
            total_seconds=time.time()-level_start,
        )
        return nxt, selected, candidates, kernels


def parse_args():
    parser = argparse.ArgumentParser(description="Generic GPU commutator reservoir search")
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--twisted-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=6); parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--p", type=int, default=3); parser.add_argument("--generator", type=int, required=True)
    parser.add_argument("--frontier-length", type=int, default=1)
    parser.add_argument("--target-length", type=int, default=127)
    parser.add_argument("--bucket-size", type=int, default=10000)
    parser.add_argument("--use-best", type=int, default=22000)
    parser.add_argument("--save-best", type=int, default=10000)
    parser.add_argument("--degree-window", type=int, default=1021)
    parser.add_argument("--shard-count", type=int, default=1); parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--expansion-chunk", type=int, default=10000)
    parser.add_argument("--matmul-chunk", type=int, default=1000)
    parser.add_argument("--boundary-margin", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prepare_only or not args.twisted_cache.exists():
        build_twisted_cache(args.table, args.twisted_cache, args.generator)
        if args.prepare_only:
            return
    names = {
        "n", "r", "p", "frontier_length", "target_length", "bucket_size",
        "use_best", "save_best", "degree_window", "shard_count", "shard_index",
        "seed", "expansion_chunk", "matmul_chunk", "boundary_margin", "device",
    }
    cfg = SearchConfig(**{name: getattr(args, name) for name in names})
    CommutatorSearch(cfg, args.table, args.twisted_cache, args.output_dir, args.generator).run()


if __name__ == "__main__":
    main()
