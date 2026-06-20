"""
COMMUTATOR BRAID SEARCH: GPU-accelerated search for kernel elements
of the form [σ_i, g^{-1}] in the Burau representation of B_4.

Mathematical setup:
  - Fix a generator σ_i (i=1,2,3)
  - Build braids g in Garside normal form g = b_1 b_2 ... b_k
  - Impose avoidance: b_1 is left-minimal mod the centralizer of σ_i
    (left descent set of b_1 avoids the centralizer generators)
  - Compute Burau matrix of [σ_i, g^{-1}] = σ_i g^{-1} σ_i^{-1} g
  - Reservoir sample based on projlen of this commutator matrix

Key optimization: Define T_b = M_{σ_i} · M_b^{-1} · M_{σ_i}^{-1}.
Then if C_g = Burau([σ_i, g^{-1}]), the update rule is:
    C_{g·b} = T_b · C_g · M_b
So we track ONE matrix per braid (the commutator matrix C_g),
with two matmuls per expansion step (left by T_b, right by M_b).
"""

import os
import sys
import hashlib

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
import time
import gc
import math
import itertools
from typing import Optional

from braid_search import (
    STORAGE_DTYPE_MATRIX, STORAGE_DTYPE_WORD, STORAGE_DTYPE_LENGTH,
    COMPUTE_DTYPE_INT,
    compute_projlen_batch, is_scalar_identity_batch,
    build_expansion_indices_vectorized,
    GPUBuckets,
    load_tables_from_file,
)


def is_trivial_identity_batch(matrices: torch.Tensor, center: int) -> torch.Tensor:
    """
    Check if matrices are exactly the 3x3 identity at degree 0.

    These correspond to trivial commutators (g commutes with σ_i).
    They are dead ends for the search: if C_g = I, then
    C_{g·b} = T_b · I · M_b = T_b · M_b (same as level-1 commutator for b).
    """
    N, _, _, D = matrices.shape
    device = matrices.device

    identity = torch.zeros(3, 3, D, dtype=matrices.dtype, device=device)
    identity[0, 0, center] = 1
    identity[1, 1, center] = 1
    identity[2, 2, center] = 1

    return (matrices == identity.unsqueeze(0)).reshape(N, -1).all(dim=1)


def build_perm_compose_table(n=4) -> torch.Tensor:
    """Build lookup table: compose[a, b] = index of perm(a) ∘ perm(b)."""
    num = math.factorial(n)
    table = torch.zeros(num, num, dtype=torch.long)
    for a in range(num):
        pa = index_to_perm(a, n)
        for b in range(num):
            pb = index_to_perm(b, n)
            pc = tuple(pa[pb[i]] for i in range(n))
            table[a, b] = perm_to_index(pc, n)
    return table


def centralizer_perm_set(gen_idx: int, n: int = 4) -> set:
    """
    Get the set of permutation indices forming the centralizer's
    permutation subgroup ⟨s_j : j ∈ J⟩ where J = centralizer_generators(gen_idx).
    """
    J = centralizer_generators(gen_idx, n)
    subgroup = {0}  # identity
    queue = [0]
    while queue:
        current = queue.pop()
        perm_c = list(index_to_perm(current, n))
        for j in J:
            perm_new = perm_c.copy()
            perm_new[j], perm_new[j + 1] = perm_new[j + 1], perm_new[j]
            idx = perm_to_index(tuple(perm_new), n)
            if idx not in subgroup:
                subgroup.add(idx)
                queue.append(idx)
    return subgroup


def compute_braid_perms(
    words: torch.Tensor,
    lengths: torch.Tensor,
    compose_table: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the permutation index of each braid from its Garside word.

    words: (N, max_length) tensor of factor indices
    lengths: (N,) tensor of actual lengths
    compose_table: (24, 24) lookup table on the same device
    Returns: (N,) tensor of permutation indices
    """
    N = words.shape[0]
    max_len = words.shape[1]
    perms = torch.zeros(N, dtype=torch.long, device=words.device)  # identity

    for pos in range(max_len):
        mask = pos < lengths.long()
        if not mask.any():
            break
        factors = words[:, pos].long()
        new_perms = compose_table[perms, factors]
        perms = torch.where(mask, new_perms, perms)

    return perms


def is_scalar_antidiag_batch(matrices: torch.Tensor) -> torch.Tensor:
    """
    Check if matrices are scalar multiples of the anti-diagonal matrix
    (i.e., Burau matrix of an odd power of Delta).
    
    For 3x3: anti-diagonal positions are (0,2), (1,1), (2,0).
    All three must be equal (as polynomials), and all other entries zero.
    """
    N, _, _, D = matrices.shape
    device = matrices.device
    
    is_antidiag = torch.ones(N, dtype=torch.bool, device=device)
    
    sub_batch = 50000
    for start in range(0, N, sub_batch):
        end = min(start + sub_batch, N)
        batch = matrices[start:end]
        
        # Non-antidiagonal entries must be zero: (0,0),(0,1),(1,0),(1,2),(2,1),(2,2)
        off_zero = (
            (batch[:, 0, 0, :] == 0).all(dim=-1) &
            (batch[:, 0, 1, :] == 0).all(dim=-1) &
            (batch[:, 1, 0, :] == 0).all(dim=-1) &
            (batch[:, 1, 2, :] == 0).all(dim=-1) &
            (batch[:, 2, 1, :] == 0).all(dim=-1) &
            (batch[:, 2, 2, :] == 0).all(dim=-1)
        )
        
        # Anti-diagonal entries (0,2), (1,1), (2,0) must be equal
        ad_02 = batch[:, 0, 2, :]
        ad_11 = batch[:, 1, 1, :]
        ad_20 = batch[:, 2, 0, :]
        
        ad_equal = (
            (ad_02 == ad_11).all(dim=-1) &
            (ad_11 == ad_20).all(dim=-1)
        )
        
        # Must also be nonzero (not the zero matrix)
        any_nonzero = (ad_02 != 0).any(dim=-1)
        
        is_antidiag[start:end] = off_zero & ad_equal & any_nonzero
    
    return is_antidiag


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CommutatorConfig:
    """Configuration for commutator braid search."""
    bucket_size: int = 50000
    max_length: int = 50
    bootstrap_length: int = 5
    prime: int = 5
    degree_multiplier: int = 4
    device: str = "cuda"
    expansion_chunk_size: int = 100000
    use_best: int = 0
    matmul_chunk_size: int = 20000
    generator_index: int = 1  # Which σ_i to use (1, 2, or 3)

    @property
    def degree_window(self) -> int:
        """Centered degree window: [-center, center] stored as [0, 2*center]."""
        # We need room for both positive (from M_b) and negative (from T_b) degrees.
        # The commutator matrix has bounded degrees if it's close to the kernel.
        return 2 * self.degree_multiplier * self.max_length + 1

    @property
    def center(self) -> int:
        """Index in the degree array corresponding to degree 0."""
        return self.degree_multiplier * self.max_length


# =============================================================================
# PERMUTATION UTILITIES
# =============================================================================

def index_to_perm(index: int, n: int = 4) -> tuple:
    """Convert lexicographic index to permutation tuple."""
    available = list(range(n))
    perm = []
    for i in range(n):
        fact = math.factorial(n - 1 - i)
        q, index = divmod(index, fact)
        perm.append(available.pop(q))
    return tuple(perm)


def perm_to_index(perm: tuple, n: int = 4) -> int:
    """Convert permutation tuple to lexicographic index."""
    perm = list(perm)
    available = list(range(n))
    index = 0
    for i in range(n):
        pos = available.index(perm[i])
        index += pos * math.factorial(n - 1 - i)
        available.remove(perm[i])
    return index


def perm_inverse(perm: tuple) -> tuple:
    """Compute inverse permutation."""
    n = len(perm)
    inv = [0] * n
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def left_descent_set(perm: tuple) -> set:
    """
    Left descent set of a permutation.
    {i : perm^{-1}(i) > perm^{-1}(i+1)} for i in {0, ..., n-2}.
    """
    inv = perm_inverse(perm)
    n = len(perm)
    return {i for i in range(n - 1) if inv[i] > inv[i + 1]}


def centralizer_generators(gen_idx: int, n: int = 4) -> set:
    """
    Return the set of simple reflection indices that generate
    the parabolic subgroup centralizing σ_{gen_idx}.

    In B_n, σ_i commutes with σ_j iff |i - j| >= 2.
    The centralizer of σ_i (as a parabolic subgroup of the Coxeter group)
    is generated by {s_j : j = i or |j - i| >= 2}.

    gen_idx: 1-indexed generator (1, 2, or 3 for B_4)
    Returns: set of 0-indexed descent positions
    """
    # Convert to 0-indexed: σ_{gen_idx} corresponds to s_{gen_idx-1}
    s = gen_idx - 1  # 0-indexed simple reflection
    return {j for j in range(n - 1) if j == s or abs(j - s) >= 2}


def compute_allowed_first_factors(gen_idx: int, n: int = 4) -> list:
    """
    Compute which simple elements (permutation indices 1..n!-1) are allowed
    as the first Garside factor under the avoidance condition.

    The first factor b_1 must be left-minimal in its W_J-coset, meaning
    left_descent_set(b_1) ∩ J = ∅, where J = centralizer_generators(gen_idx).

    Returns list of permutation indices (excluding identity = 0).
    """
    J = centralizer_generators(gen_idx, n)
    num_simples = math.factorial(n)
    allowed = []

    for idx in range(1, num_simples):  # Skip identity (index 0)
        perm = index_to_perm(idx, n)
        lds = left_descent_set(perm)
        if lds.isdisjoint(J):
            allowed.append(idx)

    return allowed


# =============================================================================
# POLYNOMIAL MATRIX ARITHMETIC (for precomputing T_b)
# =============================================================================

def poly_mul_mod(a: torch.Tensor, b: torch.Tensor, p: int) -> torch.Tensor:
    """
    Multiply two 1D polynomial tensors (convolution) mod p.
    Returns tensor of length len(a) + len(b) - 1.
    """
    # Use float conv1d for speed
    a_f = a.float().unsqueeze(0).unsqueeze(0)  # (1, 1, La)
    b_f = b.float().flip(-1).unsqueeze(0).unsqueeze(0)  # (1, 1, Lb) flipped for conv
    pad = b.shape[0] - 1
    result = F.conv1d(a_f, b_f, padding=pad).squeeze(0).squeeze(0)
    return torch.round(result).long() % p


def polymat_mul_mod(A: torch.Tensor, B: torch.Tensor, p: int) -> torch.Tensor:
    """
    Multiply two polynomial matrices (3, 3, D_a) x (3, 3, D_b) mod p.
    Returns (3, 3, D_a + D_b - 1).
    """
    Da = A.shape[-1]
    Db = B.shape[-1]
    Dc = Da + Db - 1
    C = torch.zeros(3, 3, Dc, dtype=torch.long)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                prod = poly_mul_mod(A[i, k], B[k, j], p)
                C[i, j, :len(prod)] += prod
            C[i, j] %= p
    return C


def polymat_adjugate(M: torch.Tensor, p: int) -> torch.Tensor:
    """
    Compute the adjugate (classical adjoint) of a 3×3 polynomial matrix mod p.
    adj(M)[i,j] = (-1)^{i+j} * det(M with row j, col i removed)
    """
    D = M.shape[-1]
    # Each cofactor is a product of two polynomials -> degree 2*(D-1)
    Dc = 2 * D - 1
    adj = torch.zeros(3, 3, Dc, dtype=torch.long)

    for i in range(3):
        for j in range(3):
            # Minor M_{ji}: delete row j, col i
            rows = [r for r in range(3) if r != j]
            cols = [c for c in range(3) if c != i]
            # det of 2x2 = M[r0,c0]*M[r1,c1] - M[r0,c1]*M[r1,c0]
            t1 = poly_mul_mod(M[rows[0], cols[0]], M[rows[1], cols[1]], p)
            t2 = poly_mul_mod(M[rows[0], cols[1]], M[rows[1], cols[0]], p)
            maxlen = max(len(t1), len(t2))
            t1_pad = F.pad(t1, (0, maxlen - len(t1)))
            t2_pad = F.pad(t2, (0, maxlen - len(t2)))
            cofactor = (t1_pad - t2_pad) % p
            sign = (-1) ** (i + j)
            adj[i, j, :len(cofactor)] = (sign * cofactor) % p

    return adj


def polymat_det(M: torch.Tensor, p: int) -> torch.Tensor:
    """Compute determinant of a 3×3 polynomial matrix mod p."""
    # Expansion along first row
    D = M.shape[-1]
    Dc = 3 * D - 2  # max degree of det
    det = torch.zeros(Dc, dtype=torch.long)

    for j in range(3):
        rows = [1, 2]
        cols = [c for c in range(3) if c != j]
        minor = poly_mul_mod(M[1, cols[0]], M[2, cols[1]], p) - \
                poly_mul_mod(M[1, cols[1]], M[2, cols[0]], p)
        minor = minor % p
        term = poly_mul_mod(M[0, j], minor[:max(1, len(minor))], p)
        sign = (-1) ** j
        maxlen = max(len(det), len(term))
        det = F.pad(det, (0, max(0, maxlen - len(det))))
        term = F.pad(term, (0, max(0, maxlen - len(term))))
        det = (det + sign * term) % p

    return det


def polymat_inverse(M: torch.Tensor, p: int) -> tuple:
    """
    Compute M^{-1} for a 3×3 polynomial matrix mod p.

    For Burau matrices, det(M) = c * v^d (a monomial mod p).
    Returns (M_inv, degree_shift) where M_inv has non-negative indices
    and degree_shift is the offset to subtract.
    
    Actually returns (adj(M) * c_inv, -d) meaning:
    M^{-1}[i,j](v) = M_inv[i,j](v) * v^{degree_shift}
    
    We return the full inverse as a polynomial matrix with a global degree shift.
    """
    det = polymat_det(M, p)
    adj = polymat_adjugate(M, p)

    # det should be a monomial: find the single nonzero coefficient
    nonzero = torch.nonzero(det, as_tuple=True)[0]
    assert len(nonzero) == 1, f"Determinant is not a monomial! nonzero at {nonzero.tolist()}, det = {det.tolist()}"

    det_degree = nonzero[0].item()
    det_coeff = det[det_degree].item()

    # Modular inverse of the coefficient
    coeff_inv = pow(det_coeff, p - 2, p)  # Fermat's little theorem

    # M^{-1} = coeff_inv * v^{-det_degree} * adj(M)
    # We store this as: M_inv_shifted = coeff_inv * adj(M), with degree_shift = -det_degree
    M_inv = (adj * coeff_inv) % p

    return M_inv, -det_degree


def embed_in_window(poly_mat: torch.Tensor, degree_shift: int, D: int, center: int, p: int) -> torch.Tensor:
    """
    Embed a polynomial matrix into a centered degree window.
    
    poly_mat: (3, 3, D_src) with coefficients starting at degree 0
    degree_shift: global shift to apply (can be negative)
    D: total window size
    center: index corresponding to degree 0
    
    The coefficient at poly_mat[i,j,k] represents degree (k + degree_shift).
    This goes to index (center + k + degree_shift) in the output.
    """
    result = torch.zeros(3, 3, D, dtype=torch.long)
    D_src = poly_mat.shape[-1]

    for k in range(D_src):
        target_idx = center + k + degree_shift
        if 0 <= target_idx < D:
            result[:, :, target_idx] = (result[:, :, target_idx] + poly_mat[:, :, k]) % p

    return result


def precompute_twisted_matrices(
    simple_burau_raw: torch.Tensor,
    gen_idx: int,
    p: int,
    D: int,
    center: int,
    loaded_center: int,
) -> torch.Tensor:
    """
    Precompute T_b = M_{σ_i} · M_b^{-1} · M_{σ_i}^{-1} for each simple b.

    simple_burau_raw: (24, 3, 3, D_loaded) raw loaded Burau matrices (with loaded_center)
    gen_idx: which generator (1, 2, or 3)
    p: prime
    D: output degree window size
    center: center index of output window

    Returns: (24, 3, 3, D) tensor of T_b matrices in the centered window
    """
    # Identify the permutation index of σ_{gen_idx}
    # σ_1 = (1,0,2,3), σ_2 = (0,2,1,3), σ_3 = (0,1,3,2)
    sigma_perms = {
        1: (1, 0, 2, 3),
        2: (0, 2, 1, 3),
        3: (0, 1, 3, 2),
    }
    sigma_perm = sigma_perms[gen_idx]
    sigma_idx = perm_to_index(sigma_perm, 4)

    print(f"  σ_{gen_idx} = permutation {sigma_perm}, index {sigma_idx}")

    # Extract M_{σ_i} from the loaded table
    M_sigma_raw = simple_burau_raw[sigma_idx].long()

    # Trim to nonzero range
    def trim_to_nonzero(mat):
        """Find the actual nonzero range and return (trimmed, start_degree)."""
        nonzero_mask = mat.abs().sum(dim=(0, 1)) > 0
        if not nonzero_mask.any():
            return torch.zeros(3, 3, 1, dtype=torch.long), 0
        indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]
        start = indices[0].item()
        end = indices[-1].item() + 1
        true_start_degree = start - loaded_center
        return mat[:, :, start:end].clone(), true_start_degree

    M_sigma, sigma_deg_shift = trim_to_nonzero(M_sigma_raw)
    M_sigma = M_sigma % p

    # Compute M_{σ_i}^{-1}
    M_sigma_inv, sigma_inv_deg_shift = polymat_inverse(M_sigma, p)
    sigma_inv_deg_shift += sigma_deg_shift  # combine shifts

    print(f"  M_σ degree shift: {sigma_deg_shift}, M_σ^-1 degree shift: {sigma_inv_deg_shift}")

    # For each simple b, compute T_b = M_σ · M_b^{-1} · M_σ^{-1}
    num_simples = simple_burau_raw.shape[0]
    twisted = torch.zeros(num_simples, 3, 3, D, dtype=torch.long)

    for b in range(num_simples):
        M_b_raw = simple_burau_raw[b].long()
        M_b, b_deg_shift = trim_to_nonzero(M_b_raw)
        M_b = M_b % p

        if M_b.abs().sum() == 0:
            # Identity or zero matrix - T_b = M_σ · I · M_σ^{-1} = I
            # Actually for identity (b=0), T_0 = M_σ · M_σ^{-1} = I
            twisted[b, 0, 0, center] = 1
            twisted[b, 1, 1, center] = 1
            twisted[b, 2, 2, center] = 1
            continue

        # Compute M_b^{-1}
        M_b_inv, b_inv_deg_shift = polymat_inverse(M_b, p)
        b_inv_deg_shift += b_deg_shift  # combine shifts

        # T_b = M_σ · M_b^{-1} · M_σ^{-1}
        # Step 1: M_σ · M_b^{-1}
        step1 = polymat_mul_mod(M_sigma, M_b_inv, p)
        step1_shift = sigma_deg_shift + b_inv_deg_shift

        # Step 2: (M_σ · M_b^{-1}) · M_σ^{-1}
        T_b = polymat_mul_mod(step1, M_sigma_inv, p)
        T_b_shift = step1_shift + sigma_inv_deg_shift

        # Embed in the centered window
        twisted[b] = embed_in_window(T_b, T_b_shift, D, center, p)

    return twisted


# =============================================================================
# FAST POLYNOMIAL MATMUL (supports both left and right multiply)
# =============================================================================

class CommutatorFastPolyMatmul:
    """
    Precomputes and caches FFTs of both simple Burau matrices (M_b)
    and twisted inverse matrices (T_b) for maximum speed.
    
    Supports:
    - Right multiply: C · M_b  (same as original)
    - Left multiply: T_b · C   (new for commutator update)
    """

    def __init__(self, simple_burau: torch.Tensor, twisted_burau: torch.Tensor,
                 D: int, device: torch.device):
        self.D = D
        self.out_D = 2 * D - 1
        self.fft_size = 1 << (self.out_D).bit_length()
        self.device = device

        print(f"  Precomputing FFTs...")
        print(f"    D={D}, out_D={self.out_D}, fft_size={self.fft_size}")

        self.simple_burau = simple_burau.to(device)
        self.twisted_burau = twisted_burau.to(device)

        # FFTs for right multiply (by M_b)
        self.simple_burau_fft = torch.fft.rfft(
            simple_burau.float().to(device),
            n=self.fft_size, dim=-1
        )

        # FFTs for left multiply (by T_b)
        self.twisted_burau_fft = torch.fft.rfft(
            twisted_burau.float().to(device),
            n=self.fft_size, dim=-1
        )

        fft_mem = (self.simple_burau_fft.numel() + self.twisted_burau_fft.numel()) * 8 / 1e6
        print(f"    Total FFT cache: {fft_mem:.1f} MB")

    def matmul_right_batch(self, A: torch.Tensor, suffix_indices: torch.Tensor,
                           p: int, chunk_size: int = 20000) -> torch.Tensor:
        """Batch right multiply: C[n] = A[n] @ M_b[suffix[n]]."""
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

    def matmul_left_batch(self, A: torch.Tensor, suffix_indices: torch.Tensor,
                          p: int, chunk_size: int = 20000) -> torch.Tensor:
        """Batch left multiply: C[n] = T_b[suffix[n]] @ A[n]."""
        N = A.shape[0]
        C = torch.zeros(N, 3, 3, self.out_D, dtype=COMPUTE_DTYPE_INT, device=self.device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)

            A_chunk = A[start:end].float()
            chunk_indices = suffix_indices[start:end]

            A_fft = torch.fft.rfft(A_chunk, n=self.fft_size, dim=-1)
            del A_chunk

            T_fft = self.twisted_burau_fft[chunk_indices]

            # T[n] @ A[n]: left matrix is T, right matrix is A
            C_fft = torch.einsum('nikf,nkjf->nijf', T_fft, A_fft)
            del A_fft, T_fft

            C_real = torch.fft.irfft(C_fft, n=self.fft_size, dim=-1)
            del C_fft

            C_int = torch.round(C_real[..., :self.out_D]).to(COMPUTE_DTYPE_INT) % p
            del C_real

            C[start:end] = C_int
            del C_int

        return C

    def recenter_after_convolution(self, matrices: torch.Tensor, p: int) -> torch.Tensor:
        """
        After a convolution of two centered-window polynomials (each of size D,
        centered at index `center`), the output has size out_D = 2D-1,
        and degree 0 is at index 2*center. We extract D coefficients
        centered around that position to get back to the original window.
        """
        current_D = matrices.shape[-1]
        target_D = self.D
        
        if current_D <= target_D:
            pad_needed = target_D - current_D
            return F.pad(matrices, (0, pad_needed), value=0)
        
        # After convolution of two arrays centered at `center`:
        # degree 0 in the output is at index center + center = 2*center
        # We want D coefficients: [2*center - center, 2*center + center] = [center, center + D)
        center = self.D // 2
        start = center
        return matrices[..., start:start + target_D] % p

    def commutator_expand_batch(self, C_matrices: torch.Tensor,
                                suffix_indices: torch.Tensor,
                                p: int, chunk_size: int = 20000) -> torch.Tensor:
        """
        Compute C_{gb} = T_b · C_g · M_b for a batch.
        
        Two sequential matmuls: first left by T_b, then right by M_b.
        We TRUNCATE the intermediate result back to D coefficients between
        the two matmuls to avoid FFT aliasing (since the second convolution
        would exceed out_D without truncation).
        """
        N = C_matrices.shape[0]

        # Step 1: intermediate = T_b · C_g (left multiply)
        intermediate = self.matmul_left_batch(C_matrices, suffix_indices, p, chunk_size)

        # CRITICAL: truncate back to D before second matmul to avoid aliasing
        intermediate = self.recenter_after_convolution(intermediate, p)

        # Step 2: result = intermediate · M_b (right multiply)
        result = self.matmul_right_batch(intermediate, suffix_indices, p, chunk_size)
        del intermediate

        return result


# =============================================================================
# MAIN ALGORITHM
# =============================================================================

class CommutatorBraidSearch:
    """
    GPU-accelerated search for kernel elements of the form [σ_i, g^{-1}].
    
    Tracks the commutator matrix C_g = Burau([σ_i, g^{-1}]) directly,
    using the update rule C_{gb} = T_b · C_g · M_b.
    """

    def __init__(
        self,
        simple_burau: torch.Tensor,
        twisted_burau: torch.Tensor,
        valid_suffixes: torch.Tensor,
        num_valid_suffixes: torch.Tensor,
        allowed_first_factors: list,
        config: CommutatorConfig,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.D = config.degree_window
        self.center = config.center

        self.simple_burau = simple_burau.to(STORAGE_DTYPE_MATRIX).to(self.device)
        self.twisted_burau = twisted_burau.to(STORAGE_DTYPE_MATRIX).to(self.device)
        self.valid_suffixes = valid_suffixes.to(self.device)
        self.num_valid_suffixes = num_valid_suffixes.to(self.device)
        self.allowed_first_factors = allowed_first_factors

        self.fast_matmul = CommutatorFastPolyMatmul(
            simple_burau, twisted_burau, self.D, self.device
        )

        # Permutation tools for filtering trivial commutators
        self.compose_table = build_perm_compose_table(n=4).to(self.device)
        self.centralizer_perms = centralizer_perm_set(config.generator_index, n=4)

        self.buckets: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.kernel_braids: list[torch.Tensor] = []

    def _filter_kernel_candidates(self, words, lengths):
        """
        Filter kernel candidates: only keep braids g where π(g) is NOT in
        the centralizer's permutation subgroup. If π(g) ∈ centralizer perms,
        then g almost certainly commutes with σ_i (trivial commutator).

        Returns (nontrivial_words, num_trivial, num_nontrivial).
        """
        if len(words) == 0:
            return words, 0, 0
        perms = compute_braid_perms(words, lengths, self.compose_table)
        # Vectorized membership check
        centralizer_tensor = torch.tensor(
            sorted(self.centralizer_perms), dtype=torch.long, device=words.device
        )
        in_centralizer = (perms.unsqueeze(1) == centralizer_tensor.unsqueeze(0)).any(dim=1)
        nontrivial = ~in_centralizer
        num_trivial = in_centralizer.sum().item()
        num_nontrivial = nontrivial.sum().item()
        return words[nontrivial], num_trivial, num_nontrivial

    def initialize(self):
        """
        Initialize with all allowed first factors.
        
        For each allowed simple b_1, compute:
          C_{b_1} = T_{b_1} · I · M_{b_1} = T_{b_1} · M_{b_1}
        
        (Since C_identity = I and the first expansion is g = b_1.)
        
        Actually, C_g for g = b_1 is:
          [σ_i, b_1^{-1}] = σ_i · b_1^{-1} · σ_i^{-1} · b_1
          Burau = M_{σ_i} · M_{b_1}^{-1} · M_{σ_i}^{-1} · M_{b_1} = T_{b_1} · M_{b_1}
        """
        num_first = len(self.allowed_first_factors)
        print(f"Initializing with {num_first} allowed first factors (avoidance condition)")
        print(f"  Generator: σ_{self.config.generator_index}")
        print(f"  Allowed first factor indices: {self.allowed_first_factors}")

        # Compute C_{b_1} = T_{b_1} · M_{b_1} for each allowed first factor
        # We do this by treating identity as the "current" matrix and expanding
        identity_matrices = torch.zeros(
            num_first, 3, 3, self.D, dtype=COMPUTE_DTYPE_INT, device=self.device
        )
        for i in range(num_first):
            identity_matrices[i, 0, 0, self.center] = 1
            identity_matrices[i, 1, 1, self.center] = 1
            identity_matrices[i, 2, 2, self.center] = 1

        suffix_indices = torch.tensor(
            self.allowed_first_factors, dtype=torch.long, device=self.device
        )

        # C_{b_1} = T_{b_1} · I · M_{b_1}
        C_init = self.fast_matmul.commutator_expand_batch(
            identity_matrices, suffix_indices,
            self.config.prime, self.config.matmul_chunk_size
        )

        # Truncate to target window
        C_init = self.recenter_matrices(C_init)

        # Build word tensors
        init_words = torch.zeros(
            num_first, self.config.max_length,
            dtype=STORAGE_DTYPE_WORD, device=self.device
        )
        for i, factor_idx in enumerate(self.allowed_first_factors):
            init_words[i, 0] = factor_idx

        init_lengths = torch.ones(num_first, dtype=STORAGE_DTYPE_LENGTH, device=self.device)

        # Check for kernel elements BEFORE pruning from expansion pool
        kernel_mask = is_scalar_identity_batch(C_init) | is_scalar_antidiag_batch(C_init)
        num_kernel = kernel_mask.sum().item()
        if num_kernel > 0:
            # Filter by permutation: π(g) ∉ centralizer ⟹ g doesn't commute with σ_i
            # ⟹ commutator is nontrivial ⟹ genuine kernel element
            nontrivial_words, num_trivial_k, num_nontrivial_k = \
                self._filter_kernel_candidates(
                    init_words[kernel_mask], init_lengths[kernel_mask])
            if num_trivial_k > 0:
                print(f"  Filtered {num_trivial_k} trivial commutators (π(g) in centralizer)")
            if num_nontrivial_k > 0:
                print(f"  🎉 FOUND {num_nontrivial_k} KERNEL ELEMENTS in initial factors!")
                self.kernel_braids.append(nontrivial_words.cpu())

        # Filter out identity matrices before bucketing — they are dead ends
        # for expansion: if C_g = I, then C_{g·b} = T_b · I · M_b = T_b · M_b
        # (reproduces the level-1 commutator for b alone).
        # This applies to both trivial commutators (g centralizes σ_i) and
        # genuine kernel elements (Burau(commutator) = I but braid is nontrivial).
        trivial_mask = is_trivial_identity_batch(C_init, self.center)
        num_trivial = trivial_mask.sum().item()
        if num_trivial > 0:
            print(f"  Pruned {num_trivial} identity matrices from expansion pool (dead ends)")
            keep = ~trivial_mask
            C_init = C_init[keep]
            init_words = init_words[keep]
            init_lengths = init_lengths[keep]

        # Compute initial projlens and bucket
        init_projlens = compute_projlen_batch(C_init)

        # Store in buckets
        gpu_buckets = GPUBuckets(self.config.bucket_size, self.device)
        gpu_buckets.add_chunk(
            C_init.to(STORAGE_DTYPE_MATRIX),
            init_words, init_lengths, init_projlens,
            is_bootstrap=True
        )
        self.buckets = gpu_buckets.get_buckets()

        total_kept = sum(m.shape[0] for m, _, _ in self.buckets.values())
        projlen_dist = {}
        for pl in sorted(self.buckets.keys()):
            count = self.buckets[pl][0].shape[0]
            projlen_dist[pl] = count

        print(f"  Initial projlen distribution:")
        for pl in sorted(projlen_dist.keys())[:10]:
            print(f"    projlen={pl}: {projlen_dist[pl]} braids")
        print(f"  Total initialized: {total_kept}")

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
        suffix_indices: torch.Tensor,
    ) -> tuple:
        """Expand a chunk using the commutator update rule C_{gb} = T_b · C_g · M_b."""
        num_candidates = len(braid_indices)

        parent_matrices = matrices[braid_indices]
        parent_words = words[braid_indices]
        parent_lengths = lengths[braid_indices]

        # C_{gb} = T_b · C_g · M_b (two matmuls)
        new_matrices = self.fast_matmul.commutator_expand_batch(
            parent_matrices, suffix_indices,
            self.config.prime,
            chunk_size=self.config.matmul_chunk_size
        )

        new_words = parent_words.clone()
        batch_idx = torch.arange(num_candidates, device=self.device)
        new_words[batch_idx, parent_lengths.long()] = suffix_indices.to(STORAGE_DTYPE_WORD)
        new_lengths = parent_lengths + 1

        return new_matrices, new_words, new_lengths

    def recenter_matrices(self, matrices: torch.Tensor) -> torch.Tensor:
        """Truncate/pad matrices back to the target degree window D (centered).
        
        After a convolution of two arrays of size D (centered at `center`),
        the output has size 2D-1 with degree 0 at index 2*center.
        Extract D coefficients centered around that point.
        """
        current_D = matrices.shape[-1]
        target_D = self.D

        if current_D <= target_D:
            pad_needed = target_D - current_D
            return F.pad(matrices, (0, pad_needed), value=0)

        # degree 0 in convolution output is at index 2*center
        # Extract [center, center + D) to get back to the original window
        start = self.center
        return matrices[..., start:start + target_D] % self.config.prime

    def process_level(self, level: int):
        """Process one level with commutator update rule."""
        level_start = time.time()

        is_bootstrap = (level <= self.config.bootstrap_length)
        mode = "BOOTSTRAP" if is_bootstrap else "SAMPLING"

        print(f"\n{'=' * 60}")
        print(f"Level {level} - {mode}")
        print(f"{'=' * 60}")

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
        num_trivial_total = 0
        num_trivial_comm_total = 0
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

            # Check for kernel elements BEFORE pruning identity matrices,
            # since genuine kernel elements also have C_g = scalar·I
            kernel_mask = is_scalar_identity_batch(chunk_matrices) | \
                          is_scalar_antidiag_batch(chunk_matrices)
            num_kernel = kernel_mask.sum().item()
            if num_kernel > 0:
                # Filter by permutation: only keep candidates where π(g)
                # is NOT in the centralizer subgroup (those are genuine)
                nontrivial_words, num_trivial_k, num_nontrivial_k = \
                    self._filter_kernel_candidates(
                        chunk_words[kernel_mask], chunk_lengths[kernel_mask])
                num_trivial_comm_total += num_trivial_k
                if num_nontrivial_k > 0:
                    print(f"\n  🎉 FOUND {num_nontrivial_k} KERNEL ELEMENTS! 🎉")
                    self.kernel_braids.append(nontrivial_words.cpu())

            # Prune identity matrices from expansion pool (dead ends):
            # If C_g = I, then C_{g·b} = T_b · M_b (reproduces level-1).
            trivial_mask = is_trivial_identity_batch(chunk_matrices, self.center)
            num_trivial = trivial_mask.sum().item()
            if num_trivial > 0:
                num_trivial_total += num_trivial
                keep = ~trivial_mask
                chunk_matrices = chunk_matrices[keep]
                chunk_words = chunk_words[keep]
                chunk_lengths = chunk_lengths[keep]

            chunk_projlens = compute_projlen_batch(chunk_matrices)
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

        if num_trivial_total > 0:
            print(f"  Pruned {num_trivial_total} identity matrices from expansion (dead ends)")
        if num_trivial_comm_total > 0:
            print(f"  Filtered {num_trivial_comm_total} trivial commutators (π(g) in centralizer)")
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

        print(f"  Braids kept: {total_kept} (in {len(self.buckets)} buckets, {total_bytes / 1e9:.2f} GB)")

        level_time = time.time() - level_start
        print(f"  ⚡ Timing: matmul={t_matmul_total:.2f}s, sampling={t_sample_total:.2f}s, total={level_time:.2f}s")

        return True

    def run(self):
        """Run the full commutator search algorithm."""
        self.initialize()

        total_start = time.time()
        final_level = 0

        try:
            for level in range(2, self.config.max_length + 1):
                # Start at level 2 since level 1 (first factors) is handled in initialize()
                success = self.process_level(level)
                final_level = level

                if not success:
                    break

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted at level {final_level}!")

        finally:
            total_time = time.time() - total_start

            print(f"\n{'=' * 60}")
            print("SEARCH COMPLETE" if final_level == self.config.max_length else "SEARCH STOPPED")
            print(f"{'=' * 60}")
            print(f"Generator: σ_{self.config.generator_index}")
            print(f"Final level: {final_level}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Avg time per level: {total_time / max(1, final_level):.2f}s")
            print(f"Total kernel elements: {sum(len(w) for w in self.kernel_braids)}")

        return self.kernel_braids

    def frontier_records(self, limit: int = 0) -> list[dict]:
        """Export the low-projlen final frontier for mutation-model training."""
        records = []
        seen = set()
        for projlen in sorted(self.buckets):
            _, words, lengths = self.buckets[projlen]
            words_cpu = words.cpu()
            lengths_cpu = lengths.cpu().tolist()
            for row, length in zip(words_cpu, lengths_cpu):
                factor_ids = tuple(int(value) for value in row[: int(length)].tolist())
                if factor_ids in seen:
                    continue
                seen.add(factor_ids)
                digest = hashlib.blake2b(digest_size=16)
                digest.update(bytes(factor_ids))
                records.append(
                    {
                        "depth": int(length),
                        "power": 0,
                        "factor_ids": list(factor_ids),
                        # compute_projlen_batch returns support width; exported
                        # checkpoints use max_degree-min_degree like the model.
                        "author_projlen": int(projlen) - 1,
                        "matrix_fingerprint": digest.hexdigest(),
                    }
                )
                if limit > 0 and len(records) >= limit:
                    return records
        return records


# =============================================================================
# TABLE LOADING (modified for centered degree window)
# =============================================================================

def load_tables_for_commutator(config: CommutatorConfig, table_path: str):
    """
    Load precomputed tables and compute twisted matrices.
    
    Returns: (simple_burau, twisted_burau, valid_suffixes, num_valid_suffixes, allowed_first_factors)
    """
    tables = torch.load(table_path, weights_only=True)

    assert tables['n'] == 4
    assert tables['p'] == config.prime

    loaded_burau = tables['simple_burau']
    loaded_center = tables['center']

    D = config.degree_window
    center = config.center

    print(f"Degree window: [{-center}, {center}] ({D} coefficients, centered)")

    # Load simple Burau matrices into CENTERED window
    simple_burau = torch.zeros(24, 3, 3, D, dtype=torch.long)

    for s in range(24):
        mat = loaded_burau[s]
        nonzero_mask = mat.abs().sum(dim=(0, 1)) > 0
        if not nonzero_mask.any():
            # Identity: put 1 at degree 0 (index = center)
            if s == 0:
                for i in range(3):
                    simple_burau[s, i, i, center] = 1
            continue

        nonzero_indices = torch.where(nonzero_mask)[0]
        src_start = nonzero_indices[0].item()
        src_end = nonzero_indices[-1].item() + 1

        for k in range(src_start, src_end):
            true_degree = k - loaded_center
            target_idx = center + true_degree
            if 0 <= target_idx < D:
                simple_burau[s, :, :, target_idx] = mat[:, :, k].long()

    simple_burau = simple_burau % config.prime

    # Ensure identity is correct
    if simple_burau[0].abs().sum() == 0:
        for i in range(3):
            simple_burau[0, i, i, center] = 1

    # Load valid suffixes (same as original)
    loaded_valid_suffixes = tables['valid_suffixes']
    loaded_num_valid = tables['num_valid_suffixes']

    delta_idx = tables['delta_index']
    id_idx = tables['id_index']

    valid_suffixes = loaded_valid_suffixes.clone()
    num_valid_suffixes = loaded_num_valid.clone()

    # Identity gets Delta's suffix table (standard trick)
    valid_suffixes[id_idx] = valid_suffixes[delta_idx]
    num_valid_suffixes[id_idx] = num_valid_suffixes[delta_idx]

    # Compute twisted matrices T_b = M_σ · M_b^{-1} · M_σ^{-1}
    print(f"\nPrecomputing twisted matrices T_b = M_σ · M_b^{{-1}} · M_σ^{{-1}}...")
    twisted_burau = precompute_twisted_matrices(
        loaded_burau, config.generator_index, config.prime, D, center, loaded_center
    )

    # Compute allowed first factors
    allowed_first_factors = compute_allowed_first_factors(config.generator_index, n=4)
    print(f"\nAvoidance condition for σ_{config.generator_index}:")
    J = centralizer_generators(config.generator_index, n=4)
    print(f"  Centralizer generators (0-indexed): {J}")
    print(f"  Allowed first factors: {len(allowed_first_factors)} / 23 non-identity simples")
    for idx in allowed_first_factors:
        perm = index_to_perm(idx, 4)
        lds = left_descent_set(perm)
        print(f"    #{idx}: perm={perm}, left_descent={lds}")

    return (
        simple_burau.to(STORAGE_DTYPE_MATRIX),
        twisted_burau.to(STORAGE_DTYPE_MATRIX),
        valid_suffixes,
        num_valid_suffixes,
        allowed_first_factors,
    )
