"""
MAXIMALLY OPTIMIZED braid search with:
1. Batched FFT matrix multiplication (3x fewer FFT ops)
2. Precomputed FFTs of simple Burau matrices (eliminates half of per-expansion FFTs)
3. NON-NEGATIVE DEGREES ONLY - halves memory and FFT size!

Key insight: Standard Burau representation only produces non-negative powers of v,
so we don't need to store space for negative degrees. This cuts storage in half
and reduces FFT size by ~2x.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
import time
import gc

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Algorithm parameters."""
    bucket_size: int = 50000
    max_length: int = 50
    bootstrap_length: int = 5
    prime: int = 5
    degree_multiplier: int = 4
    device: str = "cuda"
    expansion_chunk_size: int = 100000
    use_best: int = 0
    matmul_chunk_size: int = 20000
    
    @property
    def degree_window(self) -> int:
        # Only non-negative degrees [0, degree_multiplier * max_length]
        return self.degree_multiplier * self.max_length + 1


# =============================================================================
# DTYPE CONFIGURATION
# =============================================================================

STORAGE_DTYPE_MATRIX = torch.int16
STORAGE_DTYPE_WORD = torch.int32
STORAGE_DTYPE_LENGTH = torch.int32
COMPUTE_DTYPE_INT = torch.int32


# =============================================================================
# ULTRA-OPTIMIZED POLYNOMIAL MATRIX MULTIPLICATION
# =============================================================================

class FastPolyMatmul:
    """
    Precomputes and caches FFTs of simple Burau matrices for maximum speed.
    """
    
    def __init__(self, simple_burau: torch.Tensor, D: int, device: torch.device):
        self.D = D
        self.out_D = 2 * D - 1
        self.fft_size = 1 << (self.out_D).bit_length()
        self.device = device
        
        print(f"Precomputing FFTs of simple Burau matrices...")
        print(f"  D={D}, out_D={self.out_D}, fft_size={self.fft_size}")
        
        self.simple_burau = simple_burau.to(device)
        self.simple_burau_fft = torch.fft.rfft(
            simple_burau.float().to(device),
            n=self.fft_size,
            dim=-1
        )
        
        fft_mem = self.simple_burau_fft.numel() * 8 / 1e6
        print(f"  FFT cache: {fft_mem:.1f} MB")
    
    def matmul_batch(self, A: torch.Tensor, suffix_indices: torch.Tensor, 
                     p: int, chunk_size: int = 20000) -> torch.Tensor:
        """Optimized batch matrix multiply with precomputed FFTs."""
        N = A.shape[0]
        C = torch.zeros(N, 3, 3, self.out_D, dtype=COMPUTE_DTYPE_INT, device=self.device)
        
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            
            A_chunk = A[start:end].float()
            chunk_indices = suffix_indices[start:end]
            
            A_fft = torch.fft.rfft(A_chunk, n=self.fft_size, dim=-1)
            del A_chunk
            
            B_fft = self.simple_burau_fft[chunk_indices]
            
            C_fft = torch.einsum('nikf,nkjf->nijf', A_fft, B_fft)
            del A_fft, B_fft
            
            C_real = torch.fft.irfft(C_fft, n=self.fft_size, dim=-1)
            del C_fft
            
            C_int = torch.round(C_real[..., :self.out_D]).to(COMPUTE_DTYPE_INT) % p
            del C_real
            
            C[start:end] = C_int
            del C_int
        
        return C


def compute_projlen_batch(matrices: torch.Tensor) -> torch.Tensor:
    """Compute projective length for a batch of matrices."""
    N, _, _, D = matrices.shape
    device = matrices.device
    
    projlens = torch.zeros(N, dtype=torch.int32, device=device)
    
    sub_batch_size = 50000
    for start in range(0, N, sub_batch_size):
        end = min(start + sub_batch_size, N)
        batch = matrices[start:end]
        batch_N = end - start
        
        flat = batch.reshape(batch_N, -1)
        has_nonzero = (flat != 0).any(dim=-1)
        
        by_degree = batch.reshape(batch_N, 9, D)
        degree_has_nonzero = (by_degree != 0).any(dim=1)
        
        min_degrees = degree_has_nonzero.int().argmax(dim=-1)
        max_degrees = D - 1 - degree_has_nonzero.flip(dims=[-1]).int().argmax(dim=-1)
        
        batch_projlen = max_degrees - min_degrees + 1
        batch_projlen = torch.where(has_nonzero, batch_projlen, torch.zeros_like(batch_projlen))
        
        projlens[start:end] = batch_projlen
    
    return projlens


def is_scalar_identity_batch(matrices: torch.Tensor) -> torch.Tensor:
    """
    Check if matrices are scalar multiples of the identity matrix.
    
    A matrix is a scalar multiple of identity iff:
    - All off-diagonal entries are zero
    - All diagonal entries are equal (as polynomials)
    
    Returns a boolean tensor of shape (N,).
    """
    N, _, _, D = matrices.shape
    device = matrices.device
    
    is_scalar_id = torch.ones(N, dtype=torch.bool, device=device)
    
    sub_batch_size = 50000
    for start in range(0, N, sub_batch_size):
        end = min(start + sub_batch_size, N)
        batch = matrices[start:end]
        batch_N = end - start
        
        # Check off-diagonal entries are all zero
        # Off-diagonal positions: (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)
        off_diag_zero = (
            (batch[:, 0, 1, :] == 0).all(dim=-1) &
            (batch[:, 0, 2, :] == 0).all(dim=-1) &
            (batch[:, 1, 0, :] == 0).all(dim=-1) &
            (batch[:, 1, 2, :] == 0).all(dim=-1) &
            (batch[:, 2, 0, :] == 0).all(dim=-1) &
            (batch[:, 2, 1, :] == 0).all(dim=-1)
        )
        
        # Check all diagonal entries are equal
        diag_00 = batch[:, 0, 0, :]  # (batch_N, D)
        diag_11 = batch[:, 1, 1, :]
        diag_22 = batch[:, 2, 2, :]
        
        diag_equal = (
            (diag_00 == diag_11).all(dim=-1) &
            (diag_11 == diag_22).all(dim=-1)
        )
        
        is_scalar_id[start:end] = off_diag_zero & diag_equal
    
    return is_scalar_id


# =============================================================================
# VECTORIZED SUFFIX EXPANSION
# =============================================================================

def build_expansion_indices_vectorized(
    last_simples: torch.Tensor,
    num_valid_suffixes: torch.Tensor,
    valid_suffixes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (braid_index, suffix_index) pairs. FULLY VECTORIZED."""
    device = last_simples.device
    N = len(last_simples)
    
    last_simples = last_simples.long()
    
    suffix_counts = num_valid_suffixes[last_simples]
    total_expansions = suffix_counts.sum().item()
    
    if total_expansions == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device)
        )
    
    braid_indices = torch.repeat_interleave(
        torch.arange(N, device=device),
        suffix_counts
    )
    
    cumsum = torch.cumsum(suffix_counts, dim=0)
    starts = cumsum - suffix_counts
    
    global_positions = torch.arange(total_expansions, device=device)
    local_suffix_indices = global_positions - starts[braid_indices]
    
    last_simples_expanded = last_simples[braid_indices]
    suffix_indices = valid_suffixes[last_simples_expanded, local_suffix_indices].long()
    
    return braid_indices, suffix_indices


# =============================================================================
# GPU BUCKETS
# =============================================================================

class GPUBuckets:
    """Maintains reservoir-sampled buckets entirely on GPU."""
    
    def __init__(self, bucket_size: int, device: torch.device):
        self.bucket_size = bucket_size
        self.device = device
        self.data: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    
    def add_chunk(
        self,
        matrices: torch.Tensor,
        words: torch.Tensor, 
        lengths: torch.Tensor,
        projlens: torch.Tensor,
        is_bootstrap: bool
    ):
        if len(matrices) == 0:
            return
        
        matrices = matrices.to(STORAGE_DTYPE_MATRIX)
        words = words.to(STORAGE_DTYPE_WORD)
        lengths = lengths.to(STORAGE_DTYPE_LENGTH)
        
        priorities = torch.rand(len(matrices), device=self.device)
        unique_pls = torch.unique(projlens)
        
        for pl in unique_pls.tolist():
            mask = (projlens == pl)
            new_mat = matrices[mask]
            new_words = words[mask]
            new_lengths = lengths[mask]
            new_priorities = priorities[mask]
            
            if pl not in self.data:
                if is_bootstrap or len(new_mat) <= self.bucket_size:
                    self.data[pl] = (new_mat, new_words, new_lengths, new_priorities)
                else:
                    _, topk_idx = torch.topk(new_priorities, self.bucket_size, largest=False)
                    self.data[pl] = (
                        new_mat[topk_idx],
                        new_words[topk_idx],
                        new_lengths[topk_idx],
                        new_priorities[topk_idx]
                    )
            else:
                old_mat, old_words, old_lengths, old_priorities = self.data[pl]
                
                merged_mat = torch.cat([old_mat, new_mat], dim=0)
                merged_words = torch.cat([old_words, new_words], dim=0)
                merged_lengths = torch.cat([old_lengths, new_lengths], dim=0)
                merged_priorities = torch.cat([old_priorities, new_priorities], dim=0)
                
                if is_bootstrap or len(merged_mat) <= self.bucket_size:
                    self.data[pl] = (merged_mat, merged_words, merged_lengths, merged_priorities)
                else:
                    _, topk_idx = torch.topk(merged_priorities, self.bucket_size, largest=False)
                    self.data[pl] = (
                        merged_mat[topk_idx],
                        merged_words[topk_idx],
                        merged_lengths[topk_idx],
                        merged_priorities[topk_idx]
                    )
                    del merged_mat, merged_words, merged_lengths, merged_priorities
    
    def get_buckets(self) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return {
            pl: (mat, words, lengths)
            for pl, (mat, words, lengths, _) in self.data.items()
        }
    
    def total_count(self) -> int:
        return sum(mat.shape[0] for mat, _, _, _ in self.data.values())
    
    def clear(self):
        self.data.clear()


# =============================================================================
# MAIN ALGORITHM
# =============================================================================

class BraidSearchUltra:
    """
    ULTRA-OPTIMIZED GPU-accelerated search for braids with low projlen.
    """
    
    def __init__(
        self,
        simple_burau: torch.Tensor,
        valid_suffixes: torch.Tensor,
        num_valid_suffixes: torch.Tensor,
        config: Config
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        self.simple_burau = simple_burau.to(STORAGE_DTYPE_MATRIX).to(self.device)
        self.valid_suffixes = valid_suffixes.to(self.device)
        self.num_valid_suffixes = num_valid_suffixes.to(self.device)
        
        self.D = simple_burau.shape[-1]
        
        self.fast_matmul = FastPolyMatmul(simple_burau, self.D, self.device)
        
        self.buckets: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.kernel_braids: list[torch.Tensor] = []
    
    def initialize(self):
        """Start with the identity braid."""
        identity_matrix = torch.zeros(1, 3, 3, self.D, dtype=STORAGE_DTYPE_MATRIX, device=self.device)
        for i in range(3):
            identity_matrix[0, i, i, 0] = 1  # v^0 = 1 at index 0
        
        identity_word = torch.zeros(1, self.config.max_length, dtype=STORAGE_DTYPE_WORD, device=self.device)
        identity_length = torch.zeros(1, dtype=STORAGE_DTYPE_LENGTH, device=self.device)
        
        self.buckets[1] = (identity_matrix, identity_word, identity_length)
        
        print(f"Initialized with identity braid")
        print(f"Degree window: [0, {self.D-1}] ({self.D} coefficients) - NON-NEGATIVE ONLY")
        print(f"FFT size: {self.fast_matmul.fft_size}")
        print(f"Storage types: matrix={STORAGE_DTYPE_MATRIX}, word={STORAGE_DTYPE_WORD}")
        print(f"Config: bucket_size={self.config.bucket_size}, "
              f"bootstrap={self.config.bootstrap_length}, "
              f"max_length={self.config.max_length}, "
              f"chunk_size={self.config.expansion_chunk_size}, "
              f"matmul_chunk={self.config.matmul_chunk_size}, "
              f"use_best={self.config.use_best if self.config.use_best > 0 else 'all'}")
        print(f"⚡ Ultra-optimized: precomputed Burau FFTs + non-negative degrees only")
    
    def gather_level_braids(self, use_best: int = 0):
        """Gather braids from current buckets, prioritizing low projlen."""
        if not self.buckets:
            raise RuntimeError("No braids to process!")
        
        sorted_projlens = sorted(self.buckets.keys())
        
        all_matrices = []
        all_words = []
        all_lengths = []
        total_selected = 0
        
        for projlen in sorted_projlens:
            matrices, words, lengths = self.buckets[projlen]
            bucket_count = len(matrices)
            
            if use_best > 0:
                remaining = use_best - total_selected
                if remaining <= 0:
                    break
                
                if bucket_count <= remaining:
                    all_matrices.append(matrices)
                    all_words.append(words)
                    all_lengths.append(lengths)
                    total_selected += bucket_count
                else:
                    idx = torch.randperm(bucket_count, device=self.device)[:remaining]
                    all_matrices.append(matrices[idx])
                    all_words.append(words[idx])
                    all_lengths.append(lengths[idx])
                    total_selected += remaining
                    break
            else:
                all_matrices.append(matrices)
                all_words.append(words)
                all_lengths.append(lengths)
        
        matrices = torch.cat(all_matrices, dim=0).to(COMPUTE_DTYPE_INT)
        words = torch.cat(all_words, dim=0)
        lengths = torch.cat(all_lengths, dim=0)
        
        batch_idx = torch.arange(len(lengths), device=self.device)
        last_pos = torch.clamp(lengths - 1, min=0).long()
        last_simples = words[batch_idx, last_pos].long()
        last_simples = torch.where(lengths > 0, last_simples, torch.zeros_like(last_simples))
        
        return matrices, words, lengths, last_simples
    
    def expand_and_multiply_chunk(
        self,
        matrices: torch.Tensor,
        words: torch.Tensor,
        lengths: torch.Tensor,
        braid_indices: torch.Tensor,
        suffix_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand a chunk using PRECOMPUTED FFTs of simple Burau matrices."""
        num_candidates = len(braid_indices)
        
        parent_matrices = matrices[braid_indices]
        parent_words = words[braid_indices]
        parent_lengths = lengths[braid_indices]
        
        new_matrices = self.fast_matmul.matmul_batch(
            parent_matrices, 
            suffix_indices,
            self.config.prime,
            chunk_size=self.config.matmul_chunk_size
        )
        
        new_words = parent_words.clone()
        batch_idx = torch.arange(num_candidates, device=self.device)
        new_words[batch_idx, parent_lengths.long()] = suffix_indices.to(STORAGE_DTYPE_WORD)
        new_lengths = parent_lengths + 1
        
        return new_matrices, new_words, new_lengths
    
    def recenter_matrices(self, matrices: torch.Tensor) -> torch.Tensor:
        """Trim matrices back to target degree window (from right only)."""
        current_D = matrices.shape[-1]
        target_D = self.D
        
        if current_D <= target_D:
            pad_needed = target_D - current_D
            return F.pad(matrices, (0, pad_needed), value=0)
        
        return matrices[..., :target_D]
    
    def process_level(self, level: int):
        """Process one level with ultra-fast matrix multiplication."""
        level_start = time.time()
        
        is_bootstrap = (level <= self.config.bootstrap_length)
        mode = "BOOTSTRAP" if is_bootstrap else "SAMPLING"
        
        print(f"\n{'='*60}")
        print(f"Level {level} - {mode}")
        print(f"{'='*60}")
        
        use_best_limit = 0 if is_bootstrap else self.config.use_best
        matrices, words, lengths, last_simples = self.gather_level_braids(use_best=use_best_limit)
        num_starting = len(matrices)
        print(f"  Starting braids: {num_starting}")
        
        braid_indices, suffix_indices = build_expansion_indices_vectorized(
            last_simples, self.num_valid_suffixes, self.valid_suffixes
        )
        num_candidates = len(braid_indices)
        print(f"  Candidates to generate: {num_candidates}")
        
        if num_candidates == 0:
            print("  No candidates! Algorithm terminates.")
            return False
        
        chunk_size = self.config.expansion_chunk_size
        num_chunks = (num_candidates + chunk_size - 1) // chunk_size
        
        if num_chunks > 1:
            print(f"  Processing in {num_chunks} chunks...")
        
        gpu_buckets = GPUBuckets(self.config.bucket_size, self.device)
        
        t_matmul_total = 0.0
        t_sample_total = 0.0
        projlen_counts: dict[int, int] = {}
        
        for chunk_idx in range(num_chunks):
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_candidates)
            
            chunk_braid_idx = braid_indices[start:end]
            chunk_suffix_idx = suffix_indices[start:end]
            
            t0 = time.time()
            chunk_matrices, chunk_words, chunk_lengths = self.expand_and_multiply_chunk(
                matrices, words, lengths,
                chunk_braid_idx, chunk_suffix_idx
            )
            chunk_matrices = self.recenter_matrices(chunk_matrices)
            t_matmul_total += time.time() - t0
            
            chunk_projlens = compute_projlen_batch(chunk_matrices)
            
            # Check for true kernel elements (scalar multiples of identity)
            kernel_mask = is_scalar_identity_batch(chunk_matrices)
            num_kernel = kernel_mask.sum().item()
            if num_kernel > 0:
                print(f"\n  🎉 FOUND {num_kernel} KERNEL ELEMENTS! 🎉")
                self.kernel_braids.append(chunk_words[kernel_mask].cpu())
            
            unique_pls, counts = torch.unique(chunk_projlens, return_counts=True)
            for pl, c in zip(unique_pls.tolist(), counts.tolist()):
                projlen_counts[pl] = projlen_counts.get(pl, 0) + c
            
            t0 = time.time()
            gpu_buckets.add_chunk(
                chunk_matrices, chunk_words, chunk_lengths, chunk_projlens,
                is_bootstrap
            )
            t_sample_total += time.time() - t0
            
            del chunk_matrices, chunk_words, chunk_lengths, chunk_projlens
        
        del matrices, words, lengths, last_simples
        del braid_indices, suffix_indices
        
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"  Projlen distribution:")
        for pl in sorted(projlen_counts.keys())[:10]:
            print(f"    projlen={pl}: {projlen_counts[pl]} braids")
        
        self.buckets = gpu_buckets.get_buckets()
        
        total_kept = sum(m.shape[0] for m, _, _ in self.buckets.values())
        
        total_bytes = 0
        for mat, wrds, lens in self.buckets.values():
            total_bytes += mat.numel() * mat.element_size()
            total_bytes += wrds.numel() * wrds.element_size()
            total_bytes += lens.numel() * lens.element_size()
        
        print(f"  Braids kept: {total_kept} (in {len(self.buckets)} buckets, {total_bytes/1e9:.2f} GB)")
        
        level_time = time.time() - level_start
        print(f"  ⚡ Timing: matmul={t_matmul_total:.2f}s, sampling={t_sample_total:.2f}s, total={level_time:.2f}s")
        
        return True
    
    def run(self):
        """Run the full search algorithm."""
        self.initialize()
        
        total_start = time.time()
        final_level = 0
        
        try:
            for level in range(1, self.config.max_length + 1):
                success = self.process_level(level)
                final_level = level
                
                if not success:
                    break
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted at level {final_level}!")
        
        finally:
            total_time = time.time() - total_start
            
            print(f"\n{'='*60}")
            print("SEARCH COMPLETE" if final_level == self.config.max_length else "SEARCH STOPPED")
            print(f"{'='*60}")
            print(f"Final level: {final_level}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Avg time per level: {total_time / max(1, final_level):.2f}s")
            print(f"Total kernel elements: {sum(len(w) for w in self.kernel_braids)}")
        
        return self.kernel_braids


# =============================================================================
# TABLE LOADING
# =============================================================================

def load_tables_from_file(config: Config, table_path: str):
    """Load precomputed tables from .pt file."""
    tables = torch.load(table_path, weights_only=True)
    
    assert tables['n'] == 4
    assert tables['p'] == config.prime
    
    loaded_burau = tables['simple_burau']
    loaded_center = tables['center']
    
    D = config.degree_window
    
    simple_burau = torch.zeros(24, 3, 3, D, dtype=STORAGE_DTYPE_MATRIX)
    
    for s in range(24):
        mat = loaded_burau[s]
        nonzero_mask = mat.abs().sum(dim=(0, 1)) > 0
        if not nonzero_mask.any():
            continue
            
        nonzero_indices = torch.where(nonzero_mask)[0]
        src_start = nonzero_indices[0].item()
        src_end = nonzero_indices[-1].item() + 1
        
        min_degree = src_start - loaded_center
        max_degree = src_end - 1 - loaded_center
        
        assert min_degree >= 0, f"Simple {s} has negative degree {min_degree}!"
        
        dst_start = min_degree
        dst_end = max_degree + 1
        
        if dst_end <= D:
            simple_burau[s, :, :, dst_start:dst_end] = mat[:, :, src_start:src_end].to(STORAGE_DTYPE_MATRIX)
        else:
            usable_end = D
            src_usable_end = src_start + (usable_end - dst_start)
            simple_burau[s, :, :, dst_start:usable_end] = mat[:, :, src_start:src_usable_end].to(STORAGE_DTYPE_MATRIX)
    
    loaded_valid_suffixes = tables['valid_suffixes']
    loaded_num_valid = tables['num_valid_suffixes']
    
    delta_idx = tables['delta_index']
    id_idx = tables['id_index']
    
    valid_suffixes = loaded_valid_suffixes.clone()
    num_valid_suffixes = loaded_num_valid.clone()
    
    valid_suffixes[id_idx] = valid_suffixes[delta_idx]
    num_valid_suffixes[id_idx] = num_valid_suffixes[delta_idx]
    
    print(f"Loaded tables from {table_path}")
    print(f"  Degree window: [0, {D-1}] ({D} coefficients) - NON-NEGATIVE ONLY")
    
    return simple_burau, valid_suffixes, num_valid_suffixes


BraidSearch = BraidSearchUltra
