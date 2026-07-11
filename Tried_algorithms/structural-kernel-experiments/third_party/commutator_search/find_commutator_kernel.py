#!/usr/bin/env python3
"""
Find kernel elements of the form [σ_i, g^{-1}] for various primes p.

This searches for braids g such that the commutator [σ_i, g^{-1}] is in the
kernel of the Burau representation mod p. The key insight is that this commutator
is invariant under left multiplication of g by elements centralizing σ_i, so we
impose an avoidance condition on the first Garside factor to eliminate redundancy.

Usage examples:
    python find_commutator_kernel.py --p 5 --gen 1 --bucket-size 100000 --use-best 50000 --max-length 60
    python find_commutator_kernel.py --p 7 --gen 2 --bucket-size 200000 --max-length 100 --device cuda
    python find_commutator_kernel.py --p 5 --gen 1 --bucket-size 4000 --max-length 30  # Quick test
"""

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PEYL_ROOT = PROJECT_ROOT / "third_party" / "braids_project"
sys.path.insert(0, str(PEYL_ROOT))

import torch

from commutator_braid_search import (
    CommutatorConfig,
    CommutatorBraidSearch,
    load_tables_for_commutator,
    index_to_perm,
)

# For verification
try:
    from peyl.braid import GNF
    from peyl.jonesrep import JonesCellRep
    from peyl import polymat
    PEYL_AVAILABLE = True
except ImportError:
    PEYL_AVAILABLE = False
    print("WARNING: peyl not available, skipping verification")


def verify_commutator_kernel(word_list, gen_idx, n=4, r=1, p=2):
    """
    Verify that [σ_{gen_idx}, g^{-1}] is in the kernel.
    
    word_list: list of Garside factor indices for g
    gen_idx: which generator (1, 2, or 3)
    """
    if not PEYL_AVAILABLE:
        return True, "Verification skipped (peyl not available)"
    
    if not word_list:
        return False, "Empty word"
    
    try:
        # Build g from Garside factors
        g = GNF(n=n, power=0, factors=tuple(word_list))
        
        # Build σ_i as a simple Artin generator
        # σ_1 = (1,0,2,3), σ_2 = (0,2,1,3), σ_3 = (0,1,3,2)
        sigma_factors = {1: (6,), 2: (2,), 3: (1,)}  # Permutation indices
        sigma = GNF(n=n, power=0, factors=sigma_factors[gen_idx])
        
        # Compute [σ_i, g^{-1}] = σ_i * g^{-1} * σ_i^{-1} * g
        g_inv = g.inv()
        sigma_inv = sigma.inv()
        commutator = sigma * g_inv * sigma_inv * g
        if commutator == GNF.identity(n):
            return False, "Trivial commutator in the braid group"
        
        # Evaluate Burau representation
        rep = JonesCellRep(n=n, r=r, p=p)
        result = rep.polymat_evaluate_braid(commutator)
        if p > 0:
            result = result % p
        
        # Check if scalar multiple of identity
        diag_poly = result[0, 0, :]
        
        is_scalar_id = True
        for i in range(3):
            for j in range(3):
                if i == j:
                    if not np.array_equal(result[i, j, :], diag_poly):
                        is_scalar_id = False
                        break
                else:
                    if np.any(result[i, j, :] != 0):
                        is_scalar_id = False
                        break
            if not is_scalar_id:
                break
        
        if is_scalar_id:
            nonzero = np.where(diag_poly != 0)[0]
            if len(nonzero) == 0:
                return False, "Evaluates to 0"
            if len(nonzero) == 1:
                coeff = int(diag_poly[nonzero[0]])
                deg = int(nonzero[0])
                return True, f"Kernel element! [σ_{gen_idx}, g^{{-1}}] = {coeff}·v^{deg} · I"
            return False, f"Multiple nonzero degrees on diagonal: {nonzero.tolist()}"
        
        # Check anti-diagonal (odd Delta power)
        antidiag = result[0, 2, :]
        is_antidiag = True
        
        if not np.array_equal(result[1, 1, :], antidiag):
            is_antidiag = False
        if not np.array_equal(result[2, 0, :], antidiag):
            is_antidiag = False
        
        if is_antidiag:
            for i, j in [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]:
                if np.any(result[i, j, :] != 0):
                    is_antidiag = False
                    break
        
        if is_antidiag:
            nonzero = np.where(antidiag != 0)[0]
            if len(nonzero) == 1:
                return True, f"Kernel element! (Delta^odd form)"
        
        return False, "Not a scalar multiple of identity or anti-diagonal"
        
    except Exception as e:
        return False, f"Verification error: {e}"


def find_commutator_kernel(
    p=5,
    gen_idx=1,
    bucket_size=4000,
    bootstrap_length=4,
    max_length=50,
    device="cuda",
    chunk_size=50000,
    use_best=0,
    degree_multiplier=2,
    matmul_chunk_size=8000,
    frontier_limit=30000,
    seed=1,
    max_witnesses_to_print=20,
):
    """Search for kernel elements of the form [σ_i, g^{-1}]."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    config = CommutatorConfig(
        bucket_size=bucket_size,
        max_length=max_length,
        bootstrap_length=bootstrap_length,
        prime=p,
        degree_multiplier=degree_multiplier,
        device=device,
        expansion_chunk_size=chunk_size,
        use_best=use_best,
        matmul_chunk_size=matmul_chunk_size,
        generator_index=gen_idx,
    )
    
    print("=" * 60)
    print(f"COMMUTATOR SEARCH: [σ_{gen_idx}, g^{{-1}}] for p={p}")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Bucket size: {config.bucket_size}")
    print(f"Max length: {config.max_length}")
    print(f"Bootstrap length: {config.bootstrap_length}")
    print(f"Prime: {config.prime}")
    print(f"Generator: σ_{config.generator_index}")
    print(f"Degree window: [{-config.center}, {config.center}] ({config.degree_window} coeffs, centered)")
    print(f"Use best: {config.use_best if config.use_best > 0 else 'unlimited'}")
    print(f"Expansion chunk size: {config.expansion_chunk_size}")
    print(f"Matmul chunk size: {config.matmul_chunk_size}")
    print()
    
    # Find table path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(script_dir, "precomputed_tables", f"tables_B4_r1_p{p}.pt"),
        os.path.join(os.path.dirname(script_dir), "precomputed_tables", f"tables_B4_r1_p{p}.pt"),
        os.path.join(script_dir, f"tables_B4_r1_p{p}.pt"),
        f"tables_B4_r1_p{p}.pt",
    ]
    
    table_path = None
    for path in possible_paths:
        if os.path.exists(path):
            table_path = path
            break
    
    if table_path is None:
        print(f"ERROR: Could not find table file for p={p}")
        print(f"Searched in: {possible_paths}")
        return None
    
    print(f"Loading tables from {table_path}...")
    try:
        simple_burau, twisted_burau, valid_suffixes, num_valid_suffixes, allowed_first_factors = \
            load_tables_for_commutator(config, table_path=table_path)
    except FileNotFoundError:
        print(f"ERROR: Table file not found at {table_path}")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Verify identity at center
    center = config.center
    assert simple_burau[0, 0, 0, center] == 1, f"Identity check failed at center={center}"
    assert simple_burau[0, 1, 1, center] == 1, "Identity check failed"
    assert simple_burau[0, 2, 2, center] == 1, "Identity check failed"
    print(f"\n✓ Identity matrix verified (at degree 0, index {center})")
    
    # Run search
    search = CommutatorBraidSearch(
        simple_burau, twisted_burau,
        valid_suffixes, num_valid_suffixes,
        allowed_first_factors, config
    )
    kernel_braids = search.run()
    frontier = search.frontier_records(limit=frontier_limit)
    
    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    verified = []
    if not kernel_braids:
        print("No kernel elements found.")

    printed_witnesses = 0
    printed_rejections = 0
    trivial_verified = 0
    rejected_candidates = 0

    for batch_idx, batch in enumerate(kernel_braids):
        print(f"\nBatch {batch_idx}: {len(batch)} candidates")
        
        for i, word_tensor in enumerate(batch):
            word_list = [w.item() for w in word_tensor]
            while word_list and word_list[-1] == 0:
                word_list.pop()
            
            if not word_list:
                continue
            
            is_kernel, msg = verify_commutator_kernel(word_list, gen_idx, p=p)
            
            if is_kernel:
                verified.append(word_list)
                if printed_witnesses < max_witnesses_to_print:
                    printed_witnesses += 1
                    print(f"\n  🎉 KERNEL ELEMENT #{len(verified)} 🎉")
                    print(f"    g factors: {word_list}")
                    print(f"    g length: {len(word_list)}")
                    print(f"    {msg}")

                    # Show the permutations
                    perms = [index_to_perm(idx) for idx in word_list]
                    print(f"    g permutations: {perms[:5]}{'...' if len(perms) > 5 else ''}")

                    if PEYL_AVAILABLE:
                        g = GNF(n=4, power=0, factors=tuple(word_list))
                        sigma_factors = {1: (6,), 2: (2,), 3: (1,)}
                        sigma = GNF(n=4, power=0, factors=sigma_factors[gen_idx])
                        comm = sigma * g.inv() * sigma.inv() * g
                        print(f"    [σ_{gen_idx}, g^{{-1}}] Artin word: {comm.magma_artin_word()}")
            elif printed_rejections < 20:
                printed_rejections += 1
                print(f"  Candidate {i}: {word_list[:8]}{'...' if len(word_list) > 8 else ''} - {msg}")
            if not is_kernel:
                rejected_candidates += 1
                if "Trivial commutator" in msg:
                    trivial_verified += 1
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print("=" * 60)
    print(f"Generator: σ_{gen_idx}")
    print(f"Prime: p={p}")
    print(f"Total candidates with projlen=1: {sum(len(b) for b in kernel_braids)}")
    print(f"Verified kernel elements: {len(verified)}")
    print(f"Rejected kernel candidates: {rejected_candidates}")
    print(f"Rejected exact trivial commutators: {trivial_verified}")
    
    if verified:
        print(f"\n✓ SUCCESS! Found {len(verified)} kernel elements of the form [σ_{gen_idx}, g^{{-1}}]")
    
    return {
        "format": "commutator-reservoir-checkpoint-v2",
        "metadata": {
            "p": p,
            "n": 4,
            "r": 1,
            "objective": "commutator_projlen",
            "generator_index": gen_idx,
            "bucket_size": bucket_size,
            "use_best": use_best,
            "bootstrap_depth": bootstrap_length,
            "target_depth": max_length,
            "actual_depth": max_length,
            "degree_multiplier": degree_multiplier,
            "seed": seed,
        },
        "verified_witnesses": verified,
        "rejected_candidates": rejected_candidates,
        "rejected_exact_trivial_commutators": trivial_verified,
        "candidates": frontier,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search for commutator kernel elements [σ_i, g^{-1}]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --p 5 --gen 1 --bucket-size 100000 --use-best 50000 --max-length 60
  %(prog)s --p 7 --gen 2 --bucket-size 200000 --max-length 100 --device cuda
  %(prog)s --p 5 --gen 1 --bucket-size 4000 --max-length 30   # Quick test

Centralizer avoidance conditions:
  σ_1: centralizer generated by {s_1, s_3} -> avoids descents {0, 2}
  σ_2: centralizer generated by {s_2}      -> avoids descent {1}
  σ_3: centralizer generated by {s_1, s_3} -> avoids descents {0, 2}
        """
    )
    
    parser.add_argument("--p", "-p", type=int, default=5,
                        help="Prime for the representation (default: 5)")
    
    parser.add_argument("--gen", "-g", type=int, default=1, choices=[1, 2, 3],
                        help="Generator index for σ_i (1, 2, or 3, default: 1)")
    
    parser.add_argument("--bucket-size", "-b", type=int, default=4000,
                        help="Number of braids to keep per projlen bucket")
    
    parser.add_argument("--bootstrap-length", "-l", type=int, default=5,
                        help="Length of initial exhaustive search")
    
    parser.add_argument("--max-length", "-m", type=int, default=50,
                        help="Maximum braid length to search")
    
    parser.add_argument("--device", "-d", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use")
    
    parser.add_argument("--chunk-size", "-c", type=int, default=50000,
                        help="Max candidates per expansion chunk")
    
    parser.add_argument("--use-best", "-u", type=int, default=0,
                        help="Max braids to expand per level")
    
    parser.add_argument("--degree-multiplier", type=int, default=2,
                        help="Degree window = 2 * multiplier * max_length + 1")
    
    parser.add_argument("--matmul-chunk", type=int, default=20000,
                        help="Chunk size for batched FFT matmul")

    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON(.gz) checkpoint with final frontier")
    parser.add_argument("--frontier-limit", type=int, default=30000,
                        help="Maximum final-frontier candidates to export")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for reservoir sampling")
    parser.add_argument("--max-witnesses-to-print", type=int, default=20,
                        help="Maximum verified witnesses to print in the log")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    verified = find_commutator_kernel(
        p=args.p,
        gen_idx=args.gen,
        bucket_size=args.bucket_size,
        bootstrap_length=args.bootstrap_length,
        max_length=args.max_length,
        device=args.device,
        chunk_size=args.chunk_size,
        use_best=args.use_best,
        degree_multiplier=args.degree_multiplier,
        matmul_chunk_size=args.matmul_chunk,
        frontier_limit=args.frontier_limit,
        seed=args.seed,
        max_witnesses_to_print=args.max_witnesses_to_print,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".gz":
            with gzip.open(output, "wt", encoding="utf-8") as handle:
                json.dump(verified, handle)
        else:
            output.write_text(json.dumps(verified, indent=2), encoding="utf-8")
        print(f"Wrote {output}")
