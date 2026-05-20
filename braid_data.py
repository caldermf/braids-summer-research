from itertools import permutations
import random
N = 4
perms = list(permutations(range(N)))


def _validate_perm(perm):
    """Return perm as a tuple and ensure it is a valid permutation of 0..n-1."""
    perm = tuple(perm)
    n = len(perm)
    if set(perm) != set(range(n)):
        raise ValueError(f"Not a valid permutation of 0..{n - 1}: {perm}")
    return perm

class GarsideFactor:
    def __init__(self, perm):
        self.perm = _validate_perm(perm)

    def left_descent(self):
        """
        Left descent set:
        {i | w^{-1}(i) > w^{-1}(i+1)}, with 0-based i in [0, n-2].
        """
        n = len(self.perm)
        inv = [0] * n
        for pos, value in enumerate(self.perm):
            inv[value] = pos
        return {i for i in range(n - 1) if inv[i] > inv[i + 1]}

    def right_descent(self):
        """
        Right descent set:
        {i | w(i) > w(i+1)}, with 0-based i in [0, n-2].
        """
        return {i for i in range(len(self.perm) - 1) if self.perm[i] > self.perm[i + 1]}

    def artin_factors(self):
        """
        Return a reduced Artin word for self.perm as adjacent transpositions.

        Output is a list of 0-based generator indices i, representing s_i that
        swaps positions i and i+1. Applying the list left-to-right to the
        identity permutation produces self.perm.
        """
        n = len(self.perm)

        # Convert perm to identity by right-multiplying adjacent transpositions.
        # Reverse that word to get a decomposition from identity to perm.
        working = list(self.perm)
        to_identity = []
        for target in range(n - 1, -1, -1):
            pos = working.index(target)
            while pos < target:
                working[pos], working[pos + 1] = working[pos + 1], working[pos]
                to_identity.append(pos)
                pos += 1

        return list(reversed(to_identity))


class GNF:
    """
    Garside normal form container:
    sigma = Delta^d * w1 * ... * w_ell

    Conditions enforced here:
    - d is an integer
    - all factors are permutations in S_n for the same n
    - ell >= 1
    - w1 != w0 (modeled as: first factor is not Delta permutation)
    - w_ell != e (last factor is not identity permutation)
    - R(w_k) superset L(w_{k+1}) for each adjacent pair
    """

    def __init__(self, d, factors):
        if not isinstance(d, int):
            raise TypeError("d must be an integer")
        self.d = d
        self.factors = [f if isinstance(f, GarsideFactor) else GarsideFactor(f) for f in factors]
        if not self.factors:
            raise ValueError("GNF requires at least one factor")

        self.n = len(self.factors[0].perm)
        for f in self.factors:
            if len(f.perm) != self.n:
                raise ValueError("All factors must lie in the same symmetric group S_n")

        self._validate_normal_form_conditions()

    @staticmethod
    def identity_perm(n):
        return tuple(range(n))

    @staticmethod
    def delta_perm(n):
        # Longest permutation in S_n: i -> n-1-i.
        return tuple(range(n - 1, -1, -1))

    def _validate_normal_form_conditions(self):
        first = self.factors[0]
        last = self.factors[-1]

        if first.perm == self.delta_perm(self.n):
            raise ValueError("Invalid GNF: w1 must not equal w0 (Delta)")
        if last.perm == self.identity_perm(self.n):
            raise ValueError("Invalid GNF: w_ell must not equal identity")

        for k in range(len(self.factors) - 1):
            left = self.factors[k]
            right = self.factors[k + 1]
            if not left.right_descent().issuperset(right.left_descent()):
                raise ValueError(
                    f"Invalid GNF at pair {k}, {k+1}: "
                    "R(w_k) must contain L(w_{k+1})"
                )

    @property
    def garside_length(self):
        return len(self.factors)

    def prefix(self, k):
        """
        Return the Garside prefix Delta^d * w1 * ... * w_k.
        k is 1-based and must satisfy 1 <= k <= ell.
        """
        ell = self.garside_length
        if k < 1 or k > ell:
            raise ValueError(f"k must be between 1 and {ell}")
        return GNF(self.d, self.factors[:k])

    def can_append_suffix(self, u):
        """
        True if u is a valid Garside suffix candidate for concatenation:
        R(w_ell) superset L(u).
        """
        u = u if isinstance(u, GarsideFactor) else GarsideFactor(u)
        if len(u.perm) != self.n:
            return False
        return self.factors[-1].right_descent().issuperset(u.left_descent())

    def append_suffix(self, u):
        """
        Return the concatenated normal form when R(w_ell) superset L(u).
        """
        u = u if isinstance(u, GarsideFactor) else GarsideFactor(u)
        if len(u.perm) != self.n:
            raise ValueError("Suffix must be in same S_n")
        if not self.can_append_suffix(u):
            raise ValueError("Not a Garside suffix: need R(w_ell) superset L(u)")
        return GNF(self.d, self.factors + [u])

    def __repr__(self):
        perms = [f.perm for f in self.factors]
        return f"GNF(d={self.d}, factors={perms})"


def _perm_from_adjacent_word(word, n):
    """
    Apply a 0-based adjacent-transposition word left-to-right to the identity.
    """
    perm = list(range(n))
    for idx in word:
        if idx < 0 or idx >= n - 1:
            raise ValueError(f"Adjacent-transposition index must be in 0..{n - 2}")
        perm[idx], perm[idx + 1] = perm[idx + 1], perm[idx]
    return tuple(perm)


def _tau_perm(perm):
    """
    Conjugation by Delta on a simple braid, expressed on the permutation braid.
    """
    perm = _validate_perm(perm)
    n = len(perm)
    mapped_word = [n - 2 - idx for idx in GarsideFactor(perm).artin_factors()]
    return _perm_from_adjacent_word(mapped_word, n)


def _poly_int_const(c):
    if c == 0:
        return {}
    return {0: c}


def _poly_int_monomial(coeff, exp):
    if coeff == 0:
        return {}
    return {exp: coeff}


def _poly_int_add(a, b):
    out = dict(a)
    for exp, coeff in b.items():
        out[exp] = out.get(exp, 0) + coeff
        if out[exp] == 0:
            del out[exp]
    return out


def _poly_int_mul(a, b):
    if not a or not b:
        return {}
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            exp = ea + eb
            out[exp] = out.get(exp, 0) + ca * cb
            if out[exp] == 0:
                del out[exp]
    return out


def _poly_int_matrix_eye(m):
    eye = [[{} for _ in range(m)] for _ in range(m)]
    for i in range(m):
        eye[i][i] = _poly_int_const(1)
    return eye


def _poly_int_matrix_mul(a, b):
    rows = len(a)
    mid = len(a[0])
    cols = len(b[0])
    out = [[{} for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            acc = {}
            for k in range(mid):
                term = _poly_int_mul(a[i][k], b[k][j])
                if term:
                    acc = _poly_int_add(acc, term)
            out[i][j] = acc
    return out


def _freeze_poly_matrix(poly_mat):
    return tuple(
        tuple(tuple(sorted(entry.items())) for entry in row)
        for row in poly_mat
    )


def _burau_generator_matrix_exact(n, i, inverse=False):
    """
    Reduced Burau matrix for sigma_i over Z[v, v^{-1}] in the convention
    sigma_1 -> [[-v^2, -v], [0, 1]]
    sigma_{n-1} -> [[1, 0], [-v, -v^2]]
    sigma_i -> [[1, 0, 0], [-v, -v^2, -v], [0, 0, 1]]
    """
    if n < 2:
        raise ValueError("Need n >= 2")
    if i < 1 or i > n - 1:
        raise ValueError(f"Generator index i must be in 1..{n-1}")

    m = n - 1
    mat = _poly_int_matrix_eye(m)

    one = _poly_int_const(1)
    minus_v = _poly_int_monomial(-1, 1)
    minus_v_sq = _poly_int_monomial(-1, 2)
    minus_v_inv = _poly_int_monomial(-1, -1)
    minus_v_inv_sq = _poly_int_monomial(-1, -2)

    if i == 1:
        if not inverse:
            mat[0][0] = minus_v_sq
            mat[0][1] = minus_v
        else:
            mat[0][0] = minus_v_inv_sq
            mat[0][1] = minus_v_inv
    elif i == n - 1:
        if not inverse:
            mat[m - 1][m - 2] = minus_v
            mat[m - 1][m - 1] = minus_v_sq
        else:
            mat[m - 1][m - 2] = minus_v_inv
            mat[m - 1][m - 1] = minus_v_inv_sq
    else:
        r0 = i - 2
        r1 = i - 1
        r2 = i
        mat[r0][r0] = one
        mat[r0][r1] = {}
        mat[r0][r2] = {}
        if not inverse:
            mat[r1][r0] = minus_v
            mat[r1][r1] = minus_v_sq
            mat[r1][r2] = minus_v
        else:
            mat[r1][r0] = minus_v_inv
            mat[r1][r1] = minus_v_inv_sq
            mat[r1][r2] = minus_v_inv
        mat[r2][r0] = {}
        mat[r2][r1] = {}
        mat[r2][r2] = one

    return mat


def burau_polynomial_matrix(word, n=4):
    """
    Evaluate the reduced Burau representation over Z[v, v^{-1}].

    word: iterable of signed generator indices.
      k > 0 means sigma_k, k < 0 means sigma_|k|^{-1}.
    Returns an (n-1)x(n-1) matrix with Laurent-polynomial dict entries:
      {exp: integer_coeff, ...}
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    m = n - 1
    result = _poly_int_matrix_eye(m)
    for g in word:
        if g == 0:
            raise ValueError("Generator index 0 is invalid")
        i = abs(g)
        gen_mat = _burau_generator_matrix_exact(n, i, inverse=(g < 0))
        result = _poly_int_matrix_mul(result, gen_mat)
    return result


_SIMPLE_BRAID_TABLE_CACHE = {}


def _simple_braid_tables(n):
    cached = _SIMPLE_BRAID_TABLE_CACHE.get(n)
    if cached is not None:
        return cached

    all_perms = list(permutations(range(n)))
    identity = GNF.identity_perm(n)
    delta = GNF.delta_perm(n)
    delta_word = [idx + 1 for idx in GarsideFactor(delta).artin_factors()]

    simple_words = {
        perm: [idx + 1 for idx in GarsideFactor(perm).artin_factors()]
        for perm in all_perms
    }
    simple_mats = {
        perm: burau_polynomial_matrix(simple_words[perm], n=n)
        for perm in all_perms
    }
    tau = {perm: _tau_perm(perm) for perm in all_perms}
    generator_to_perm = {
        gen: _perm_from_adjacent_word([gen - 1], n)
        for gen in range(1, n)
    }

    def candidate_words():
        yield (0, ())
        yield (1, ())

        for perm in all_perms:
            if perm not in (identity, delta):
                yield (0, (perm,))
                yield (1, (perm,))

        for left in all_perms:
            if left in (identity, delta):
                continue
            left_factor = GarsideFactor(left)
            for right in all_perms:
                if right in (identity, delta):
                    continue
                right_factor = GarsideFactor(right)
                if left_factor.right_descent().issuperset(right_factor.left_descent()):
                    yield (0, (left, right))

    normal_forms = {}
    for d, factors in candidate_words():
        word = []
        if d:
            word.extend(delta_word)
        for perm in factors:
            word.extend(simple_words[perm])
        key = _freeze_poly_matrix(burau_polynomial_matrix(word, n=n))
        previous = normal_forms.get(key)
        if previous is not None and previous != (d, factors):
            raise RuntimeError(
                f"Non-unique simple normal form candidate at n={n}: {previous} vs {(d, factors)}"
            )
        normal_forms[key] = (d, factors)

    pair_table = {}
    pair_inputs = [perm for perm in all_perms if perm != delta]
    for left in pair_inputs:
        for right in pair_inputs:
            product = _poly_int_matrix_mul(simple_mats[left], simple_mats[right])
            key = _freeze_poly_matrix(product)
            if key not in normal_forms:
                raise RuntimeError(
                    f"Could not normalize product of simple braids {left} and {right} in B_{n}"
                )
            pair_table[(left, right)] = normal_forms[key]

    cached = {
        "identity": identity,
        "delta": delta,
        "delta_word": delta_word,
        "simple_words": simple_words,
        "simple_mats": simple_mats,
        "tau": tau,
        "generator_to_perm": generator_to_perm,
        "pair_table": pair_table,
    }
    _SIMPLE_BRAID_TABLE_CACHE[n] = cached
    return cached


def positive_word_to_garside_normal_form(word, n=4):
    """
    Compute the left Garside normal form of a positive Artin word.

    Returns `(d, factor_perms)`, representing `Delta^d * w1 * ... * w_ell`.
    The factor list may be empty for a pure Delta power.
    """
    tables = _simple_braid_tables(n)
    factors = []
    d = 0

    for g in word:
        if g <= 0:
            raise ValueError("This normalizer currently expects a positive Artin word")
        if g >= n:
            raise ValueError(f"Generator index must lie in 1..{n - 1}")
        factors.append(tables["generator_to_perm"][g])

        changed = True
        while changed:
            changed = False
            for idx in range(len(factors) - 2, -1, -1):
                left = factors[idx]
                right = factors[idx + 1]
                pair_d, pair_factors = tables["pair_table"][(left, right)]
                pair_factors = list(pair_factors)
                if pair_d == 0 and pair_factors == [left, right]:
                    continue

                prefix = factors[:idx]
                suffix = factors[idx + 2:]
                if pair_d:
                    d += pair_d
                    prefix = [tables["tau"][perm] for perm in prefix]
                factors = prefix + pair_factors + suffix
                changed = True
                break

    return d, factors


def _poly_const(c, p):
    c %= p
    if c == 0:
        return {}
    return {0: c}


def _poly_monomial(coeff, exp, p):
    coeff %= p
    if coeff == 0:
        return {}
    return {exp: coeff}


def _poly_add(a, b, p):
    out = dict(a)
    for exp, coeff in b.items():
        out[exp] = (out.get(exp, 0) + coeff) % p
        if out[exp] == 0:
            del out[exp]
    return out


def _poly_mul(a, b, p):
    if not a or not b:
        return {}
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = ea + eb
            out[e] = (out.get(e, 0) + ca * cb) % p
            if out[e] == 0:
                del out[e]
    return out


def _poly_matrix_eye(m, p):
    eye = [[{} for _ in range(m)] for _ in range(m)]
    for i in range(m):
        eye[i][i] = _poly_const(1, p)
    return eye


def _poly_matrix_mul(a, b, p):
    rows = len(a)
    mid = len(a[0])
    cols = len(b[0])
    out = [[{} for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            acc = {}
            for k in range(mid):
                term = _poly_mul(a[i][k], b[k][j], p)
                if term:
                    acc = _poly_add(acc, term, p)
            out[i][j] = acc
    return out


def _burau_generator_matrix_poly(n, i, p, inverse=False):
    """
    Reduced Burau matrix for sigma_i (1-based i) as polynomial entries in v
    in the convention
    sigma_1 -> [[-v^2, -v], [0, 1]]
    sigma_{n-1} -> [[1, 0], [-v, -v^2]]
    sigma_i -> [[1, 0, 0], [-v, -v^2, -v], [0, 0, 1]]
    If inverse=True, returns sigma_i^{-1}, introducing negative exponents.
    """
    if n < 2:
        raise ValueError("Need n >= 2")
    if i < 1 or i > n - 1:
        raise ValueError(f"Generator index i must be in 1..{n-1}")

    m = n - 1
    mat = _poly_matrix_eye(m, p)

    one = _poly_const(1, p)
    minus_v = _poly_monomial(-1, 1, p)
    minus_v_sq = _poly_monomial(-1, 2, p)
    minus_v_inv = _poly_monomial(-1, -1, p)
    minus_v_inv_sq = _poly_monomial(-1, -2, p)

    # Convert generator index to reduced Burau matrix coordinates.
    # sigma_1 uses rows/cols 0..1, sigma_{n-1} uses m-2..m-1,
    # interior sigma_i uses block on (i-2, i-1, i) in 0-based coordinates.
    if i == 1:
        if not inverse:
            mat[0][0] = minus_v_sq
            mat[0][1] = minus_v
        else:
            mat[0][0] = minus_v_inv_sq
            mat[0][1] = minus_v_inv
    elif i == n - 1:
        if not inverse:
            mat[m - 1][m - 2] = minus_v
            mat[m - 1][m - 1] = minus_v_sq
        else:
            mat[m - 1][m - 2] = minus_v_inv
            mat[m - 1][m - 1] = minus_v_inv_sq
    else:
        r0 = i - 2
        r1 = i - 1
        r2 = i
        mat[r0][r0] = one
        mat[r0][r1] = {}
        mat[r0][r2] = {}
        if not inverse:
            mat[r1][r0] = minus_v
            mat[r1][r1] = minus_v_sq
            mat[r1][r2] = minus_v
        else:
            mat[r1][r0] = minus_v_inv
            mat[r1][r1] = minus_v_inv_sq
            mat[r1][r2] = minus_v_inv
        mat[r2][r0] = {}
        mat[r2][r1] = {}
        mat[r2][r2] = one

    return mat


def burau_mod_p_polynomial_matrix(word, p, n=4):
    """
    Evaluate reduced Burau representation of a braid word modulo p.

    word: iterable of signed generator indices.
      k > 0 means sigma_k, k < 0 means sigma_|k|^{-1}.
    Returns an (n-1)x(n-1) matrix with polynomial dict entries:
      {exp: coeff_mod_p, ...}
    """
    if p <= 1:
        raise ValueError("p must be >= 2")
    if n < 2:
        raise ValueError("n must be >= 2")

    m = n - 1
    result = _poly_matrix_eye(m, p)
    for g in word:
        if g == 0:
            raise ValueError("Generator index 0 is invalid")
        i = abs(g)
        gen_mat = _burau_generator_matrix_poly(n, i, p, inverse=(g < 0))
        result = _poly_matrix_mul(result, gen_mat, p)
    return result


def _poly_matrix_degree_bounds(poly_mat):
    exponents = []
    for row in poly_mat:
        for entry in row:
            exponents.extend(entry.keys())
    if not exponents:
        return 0, 0
    return min(exponents), max(exponents)


def burau_mod_p_projective_tensor(word, p, D, n=4):
    """
    Convert Burau(word) to a projectively normalized tensor of shape D x 3 x 3.

    The returned tensor stores coefficients after dividing by the smallest power
    of v dividing the whole matrix, so the minimum occupied degree is always 0.
    The second return value is that stripped global minimum degree.
    """
    if n != 4:
        raise ValueError("This tensor interface currently expects n=4 (3x3 matrices)")
    if D <= 0:
        raise ValueError("D must be positive")

    poly_mat = burau_mod_p_polynomial_matrix(word, p, n=n)
    min_exp, max_exp = _poly_matrix_degree_bounds(poly_mat)
    width = max_exp - min_exp + 1
    if width > D:
        raise ValueError(
            f"Tensor depth D={D} too small for projective support width {width} "
            f"(degree range {min_exp}..{max_exp})"
        )
    tensor = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(D)]

    for i in range(3):
        for j in range(3):
            for exp, coeff in poly_mat[i][j].items():
                shifted_exp = exp - min_exp
                tensor[shifted_exp][i][j] = coeff % p

    return tensor, min_exp


def burau_mod_p_tensor(word, p, D, n=4):
    """
    Convert Burau(word) to a projectively normalized tensor of shape D x 3 x 3.

    This returns only the normalized tensor for backward compatibility. Use
    burau_mod_p_projective_tensor(...) when the stripped minimum degree is also
    needed.
    """
    tensor, _ = burau_mod_p_projective_tensor(word, p, D, n=n)
    return tensor


def gnf_to_braid_word(gnf):
    """
    Convert a GNF object to a signed Artin word (1-based generators).
    """
    if not isinstance(gnf, GNF):
        raise TypeError("gnf must be an instance of GNF")

    n = gnf.n
    delta = GarsideFactor(GNF.delta_perm(n)).artin_factors()
    word = []
    if gnf.d >= 0:
        for _ in range(gnf.d):
            word.extend([idx + 1 for idx in delta])
    else:
        inv_delta = [-(idx + 1) for idx in reversed(delta)]
        for _ in range(-gnf.d):
            word.extend(inv_delta)

    for factor in gnf.factors:
        word.extend([idx + 1 for idx in factor.artin_factors()])
    return word


def burau_mod_p_tensor_from_gnf(gnf, p, D):
    """
    Evaluate the projectively normalized p-Burau tensor (D x 3 x 3) for a GNF
    object in B_4.
    """
    if not isinstance(gnf, GNF):
        raise TypeError("gnf must be an instance of GNF")
    if gnf.n != 4:
        raise ValueError("This tensor interface currently expects GNF in S_4")
    return burau_mod_p_tensor(gnf_to_braid_word(gnf), p, D, n=4)


def burau_mod_p_projective_tensor_from_gnf(gnf, p, D):
    """
    Evaluate the projectively normalized p-Burau tensor and its minimum degree
    for a GNF object in B_4.
    """
    if not isinstance(gnf, GNF):
        raise TypeError("gnf must be an instance of GNF")
    if gnf.n != 4:
        raise ValueError("This tensor interface currently expects GNF in S_4")
    return burau_mod_p_projective_tensor(gnf_to_braid_word(gnf), p, D, n=4)


def burau_mod_p_matches_delta_power_scalar(word, p, n=4, delta_power=None):
    """
    Check whether Burau(word) is a monomial scalar multiple of Burau(Delta^d) mod p.

    Returns a dict with:
      - `matches`: bool
      - `delta_power`: d reduced modulo 2 if inferred, otherwise the requested d
      - `scalar`: polynomial dict for the monomial scalar when `matches` is True
      - `matrix`: Burau(word) modulo p
      - `target`: Burau(Delta^d) modulo p for the tested d
    """
    if p <= 1:
        raise ValueError("p must be >= 2")
    if n < 2:
        raise ValueError("n must be >= 2")

    image = burau_mod_p_polynomial_matrix(word, p, n=n)
    delta_word = [idx + 1 for idx in GarsideFactor(GNF.delta_perm(n)).artin_factors()]

    if delta_power is None:
        candidates = [0, 1]
    else:
        if not isinstance(delta_power, int):
            raise TypeError("delta_power must be an integer")
        candidates = [delta_power]

    for d in candidates:
        target = burau_mod_p_polynomial_matrix(delta_word * d, p, n=n)
        scalar = None
        matches = True
        for i in range(n - 1):
            for j in range(n - 1):
                target_entry = target[i][j]
                image_entry = image[i][j]
                if not target_entry:
                    if image_entry:
                        matches = False
                        break
                    continue
                if scalar is None:
                    if len(target_entry) != 1 or len(image_entry) != 1:
                        matches = False
                        break
                    (target_exp, target_coeff), = target_entry.items()
                    (image_exp, image_coeff), = image_entry.items()
                    if target_coeff % p == 0:
                        matches = False
                        break
                    inv_coeff = pow(target_coeff, -1, p)
                    scalar = {(image_exp - target_exp): (image_coeff * inv_coeff) % p}
                expected = _poly_mul(scalar, target_entry, p)
                if image_entry != expected:
                    matches = False
                    break
            if not matches:
                break
        if matches and scalar is not None:
            return {
                "matches": True,
                "delta_power": d,
                "scalar": scalar,
                "matrix": image,
                "target": target,
            }

    return {
        "matches": False,
        "delta_power": delta_power,
        "scalar": None,
        "matrix": image,
        "target": None,
    }


class DataSetBuilder:
    """
    Build supervised data from random GNFs:
      input  = Burau tensor (D x 3 x 3),
      label1 = final Garside factor permutation,
      label2 = right descent set of that final factor.
    """

    def __init__(self, p, D, n=4, d_range=(0, 0), seed=None):
        if n != 4:
            raise ValueError("DataSetBuilder currently expects n=4 (3x3 Burau matrices)")
        if p <= 1:
            raise ValueError("p must be >= 2")
        if D <= 0:
            raise ValueError("D must be positive")
        d_min, d_max = d_range
        if d_min > d_max:
            raise ValueError("d_range must satisfy min <= max")

        self.p = p
        self.D = D
        self.n = n
        self.d_range = d_range
        self.rng = random.Random(seed)
        self._all_perms = list(permutations(range(n)))

    def _random_int(self, low, high):
        return self.rng.randint(low, high)

    def _random_choice(self, seq):
        return seq[self.rng.randrange(len(seq))]

    def _valid_factor_candidates(self, required_left_subset=None, exclude_delta=False, exclude_identity=False):
        candidates = []
        delta = GNF.delta_perm(self.n)
        identity = GNF.identity_perm(self.n)
        required_left_subset = required_left_subset or set()

        for perm in self._all_perms:
            if exclude_delta and perm == delta:
                continue
            if exclude_identity and perm == identity:
                continue
            factor = GarsideFactor(perm)
            if factor.right_descent().issuperset(required_left_subset):
                candidates.append(factor)
        return candidates

    def random_gnf(self, L, max_attempts=200):
        """
        Sample a random valid GNF with Garside length L.
        """
        if L <= 0:
            raise ValueError("L must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        for _ in range(max_attempts):
            d = self._random_int(self.d_range[0], self.d_range[1])

            # Build factors from right to left to enforce R(w_k) ⊇ L(w_{k+1}).
            factors = [None] * L
            last_candidates = self._valid_factor_candidates(exclude_identity=True)
            if not last_candidates:
                raise RuntimeError("No valid choices for final Garside factor")
            factors[-1] = self._random_choice(last_candidates)

            ok = True
            for k in range(L - 2, -1, -1):
                required = factors[k + 1].left_descent()
                candidates = self._valid_factor_candidates(
                    required_left_subset=required,
                    exclude_delta=(k == 0),
                    exclude_identity=False,
                )
                if not candidates:
                    ok = False
                    break
                factors[k] = self._random_choice(candidates)

            if ok:
                return GNF(d, factors)

        raise RuntimeError(f"Failed to sample valid GNF after {max_attempts} attempts")

    def sample(self, L, max_attempts=200):
        """
        Generate one training example.
        """
        gnf = self.random_gnf(L, max_attempts=max_attempts)
        tensor, min_degree = burau_mod_p_projective_tensor_from_gnf(gnf, p=self.p, D=self.D)
        final_factor = gnf.factors[-1]
        rdesc = sorted(final_factor.right_descent())

        return {
            "burau_tensor": tensor,
            "burau_min_degree": min_degree,
            "final_factor_perm": list(final_factor.perm),
            "final_factor_right_descent": rdesc,
            "gnf_d": gnf.d,
            "gnf_factors": [list(f.perm) for f in gnf.factors],
        }

    def build(self, num_samples, L, max_attempts=200):
        """
        Build a dataset list with num_samples iid random examples.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        return [self.sample(L, max_attempts=max_attempts) for _ in range(num_samples)]

# ---------------------------------------------------------------------------
# MCTS-facing braid helpers
# ---------------------------------------------------------------------------
# This section turns the lower-level braid/Burau utilities above into a small
# search interface. The Monte Carlo tree search should be able to ask:
#
#   1. What are the valid next Garside factors from this prefix?
#   2. What is the Burau image of this prefix?
#   3. What is its projective length?
#   4. Is this prefix a projective kernel candidate?
#
# Keeping these questions here makes monte_carlo_tree_search.py much simpler.

def all_simple_factor_perms(n=4, include_identity=True, include_delta=True) -> list[tuple[int,...]]:
    """
    Return all simple braid factors as permutations in S_n.

    In Garside normal form, each simple factor is the positive lift of a
    permutation. For B_4 there are 24 possible permutations.

    Parameters
    ----------
    n:
        Number of strands.
    include_identity:
        Whether to include the identity permutation.
    include_delta:
        Whether to include the longest permutation, corresponding to Delta.

    Returns
    -------
    list[tuple[int, ...]]
        Permutations in a stable order.
    """

    all_perms = list(permutations(range(n)))

    identity = GNF.identity_perm(n)
    delta = GNF.delta_perm(n)

    result = []
    for perm in all_perms:
        if not include_identity and perm == identity:
            continue
        if not include_delta and perm == delta:
            continue
        result.append(perm)

    return result

def simple_factor_id_maps(n=4):
    """
    Build stable maps between simple factor permutations and integer IDs.

    MCTS stores many nodes, so integer IDs are cheaper and easier to serialize
    than full permutation tuples. The order should match `itertools.permutations`
    so it stays compatible with the model code in `braidmod`.

    Returns
    -------
    tuple[dict[tuple[int, ...], int], dict[int, tuple[int, ...]]]
        `perm_to_id` and `id_to_perm`.
    """
    id_to_perm = {}
    perm_to_id = {}
    all_perms = all_simple_factor_perms(n)
    for i in range(len(all_perms)):
        id_to_perm[i] = all_perms[i]
        perm_to_id[all_perms[i]] = i
    return perm_to_id, id_to_perm

def valid_first_factor_ids(n=4) -> list:
    """
    Return factor IDs that are legal as the first GNF factor.

    In left Garside normal form:
      - the first factor may not be Delta
      - the final factor may not be identity

    Since a length-1 braid has the same factor as first and final, we exclude
    both identity and Delta here.
    """
    perm_to_id, id_to_perm = simple_factor_id_maps(n)
    id_list = []
    for i in perm_to_id:
        if (i == tuple(range(n - 1, -1, -1))) or (i == tuple(range(0,n))):
            continue
        else:
            id_list.append(perm_to_id[i])
    return id_list

def valid_suffix_factor_ids(last_factor_id, n=4) -> list:
    """
    Return all simple factor IDs that can legally follow `last_factor_id`.

    If the current prefix ends in w, a suffix u is valid exactly when

        R(w) contains L(u).

    We also exclude the identity as a suffix, because the final GNF factor is
    not allowed to be identity and appending identity does not move the search.

    Parameters
    ----------
    last_factor_id:
        Integer ID of the current final simple factor.
    n:
        Number of strands.

    Returns
    -------
    list[int]
        Legal next factor IDs.
    """
    perm_to_id, id_to_perm = simple_factor_id_maps(n)
    id_list = []
    for i in perm_to_id:
        if (i == tuple(range(n - 1, -1, -1))) or (i == tuple(range(0,n))):
            continue
        else:
            if (id_to_perm[last_factor_id].right_descent().issuperset(i.left_descent())):
                id_list.append(perm_to_id[i])
            else:
                continue
    return id_list

def valid_suffix_factor_ids(last_factor_id, n=4):
    """
    Return all simple factor IDs that can legally follow `last_factor_id`.

    If the current prefix ends in w, a suffix u is valid exactly when

        R(w) contains L(u).

    We exclude the identity as a suffix because appending identity does not
    change the braid and a final GNF factor cannot be identity.
    """
    perm_to_id, id_to_perm = simple_factor_id_maps(n)

    identity = GNF.identity_perm(n)
    identity_id = perm_to_id[identity]

    last_perm = id_to_perm[last_factor_id]
    last_factor = GarsideFactor(last_perm)
    allowed_right_descents = last_factor.right_descent()

    valid_ids = []

    for candidate_id, candidate_perm in id_to_perm.items():
        if candidate_id == identity_id:
            continue

        candidate_factor = GarsideFactor(candidate_perm)
        candidate_left_descents = candidate_factor.left_descent()

        if allowed_right_descents.issuperset(candidate_left_descents):
            valid_ids.append(candidate_id)

    return valid_ids

def factor_ids_to_perms(factor_ids, n=4) -> list[tuple[int,...]]:
    """
    Convert a list of factor IDs back to permutation tuples.

    This is mostly used for saving readable JSON results.
    """
    _, id_to_perm = simple_factor_id_maps(n)
    factor_perms = []
    for i in factor_ids:
        factor_perms.append(id_to_perm[i])
    return factor_perms

def factor_ids_to_gnf(factor_ids, d=0, n=4):
    """
    Convert factor IDs into a `GNF` object.

    Parameters
    ----------
    factor_ids:
        List of simple factor IDs.
    d:
        Delta power in the GNF. For the first MCTS version, this can stay 0.
    n:
        Number of strands.

    Returns
    -------
    GNF
        The corresponding Garside normal form object.
    """
    return GNF(d, factor_ids_to_perms(factor_ids, n))

def factor_ids_to_artin_word(factor_ids, d=0, n=4):
    """
    Convert factor IDs directly to a signed Artin generator word.

    This is useful for evaluating the Burau representation and for saving a
    result in a format that is easy to check independently.

    Returns
    -------
    list[int]
        Signed 1-based Artin generators, e.g. [1, 2, -1].
    """

    word = []
    delta_word = [idx + 1 for idx in GarsideFactor(GNF.delta_perm(n)).artin_factors()]

    if d >= 0:
        for _ in range(d):
            word.extend(delta_word)
    else:
        inverse_delta_word = [-gen for gen in reversed(delta_word)]
        for _ in range(-d):
            word.extend(inverse_delta_word)

    for perm in factor_ids_to_perms(factor_ids, n=n):
        word.extend([idx + 1 for idx in GarsideFactor(perm).artin_factors()])

    return word

def simple_factor_burau_table(p, n=4):
    """
    Precompute the Burau matrix of every simple factor modulo p.

    MCTS repeatedly appends one simple factor at a time. Instead of recomputing
    the simple factor matrix on every expansion, we cache all 24 simple images.

    Returns
    -------
    dict[int, matrix]
        Maps factor ID to its Burau polynomial matrix modulo p.
    """

    if p <= 1:
        raise ValueError("p must be >= 2")
    if n < 2:
        raise ValueError("n must be >= 2")
    
    _, id_to_perm = simple_factor_id_maps(n)

    burau_table = {}

    for i in id_to_perm:
        word = [idx + 1 for idx in GarsideFactor(id_to_perm[i]).artin_factors()]
        burau_table[i] = burau_mod_p_polynomial_matrix(word, p, n)

    return burau_table

def multiply_burau_matrices_mod_p(left, right, p):
    """
    Multiply two Burau polynomial matrices modulo p.

    This should delegate to the existing `_poly_matrix_mul` helper already in
    this file. Keeping this public wrapper gives MCTS a clean API and avoids
    importing underscore-prefixed functions elsewhere.
    """
    return _poly_matrix_mul(left, right, p)

def identity_burau_matrix(p, n=4):
    """
    Return the identity matrix in the same polynomial-matrix format used by
    `burau_mod_p_polynomial_matrix`.
    """
    if p <= 1:
        raise ValueError("p must be >= 2")
    if n < 2:
        raise ValueError("n must be >= 2")

    return _poly_matrix_eye(n - 1, p)

def append_factor_to_burau_matrix(current_matrix, factor_id, simple_table, p):
    """
    Right-multiply the current Burau image by one simple factor.

    Parameters
    ----------
    current_matrix:
        Burau image of the current prefix.
    factor_id:
        Integer ID of the simple factor being appended.
    simple_table:
        Output of `simple_factor_burau_table(p, n)`.
    p:
        Modulus.

    Returns
    -------
    matrix
        Burau image of the child prefix.
    """
    return multiply_burau_matrices_mod_p(current_matrix, simple_table[factor_id], p)

def polynomial_matrix_degree_bounds(poly_mat):
    """
    Public wrapper around `_poly_matrix_degree_bounds`.

    Returns `(min_degree, max_degree)`. If the matrix is zero, returns `(0, 0)`.
    """
    return _poly_matrix_degree_bounds(poly_mat)

def polynomial_matrix_projlen(poly_mat):
    """
    Compute projective length of a polynomial matrix.

    The paper uses:

        projlen(A) = deg(A) - val(A)

    where deg is the largest exponent and val is the smallest exponent appearing
    in any matrix entry.
    """
    min_degree, max_degree = polynomial_matrix_degree_bounds(poly_mat)
    return max_degree - min_degree

def polynomial_matrix_support_width(poly_mat):
    """
    Return the number of occupied degree slots after projective normalization.

    This is `projlen + 1` for a nonzero matrix. It is often more convenient for
    tensor depth checks.
    """
    return polynomial_matrix_projlen(poly_mat) + 1

def polynomial_matrix_to_projective_tensor(poly_mat, p, D, n=4):
    """
    Convert an already-computed Burau polynomial matrix to a projective tensor.

    Existing helpers evaluate from an Artin word. MCTS will already have the
    matrix at each node, so this avoids recomputing from scratch.

    Returns
    -------
    tuple[list, int]
        `(tensor, min_degree)` where tensor has shape `D x 3 x 3`.
    """
    min_exp, max_exp = polynomial_matrix_degree_bounds(poly_mat)
    width = polynomial_matrix_support_width(poly_mat)
    if width > D:
        raise ValueError(
            f"Tensor depth D={D} too small for projective support width {width} "
            f"(degree range {min_exp}..{max_exp})"
        )
    tensor = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(D)]

    for i in range(3):
        for j in range(3):
            for exp, coeff in poly_mat[i][j].items():
                shifted_exp = exp - min_exp
                tensor[shifted_exp][i][j] = coeff % p

    return tensor, min_exp

def is_projective_identity_matrix(poly_mat, p, n=4):
    """
    Check whether `poly_mat` equals c*v^k*I modulo p.

    Returns a structured dictionary so search logs can record the scalar
    coefficient and degree.
    """
    if p <= 1:
        raise ValueError("p must be >= 2")

    size = n - 1
    scalar = None

    for i in range(size):
        for j in range(size):
            entry = poly_mat[i][j]

            if i != j:
                # Off-diagonal entries must be zero.
                if entry:
                    return {
                        "matches": False,
                        "kernel_type": None,
                        "scalar": None,
                    }
                continue

            # Diagonal entries must be exactly one monomial.
            if len(entry) != 1:
                return {
                    "matches": False,
                    "kernel_type": None,
                    "scalar": None,
                }

            # Extract the only term in the diagonal polynomial.
            exponent, coeff = next(iter(entry.items()))
            coeff %= p

            if coeff == 0:
                return {
                    "matches": False,
                    "kernel_type": None,
                    "scalar": None,
                }

            current_scalar = {exponent: coeff}

            # First diagonal entry sets the scalar c*v^k.
            if scalar is None:
                scalar = current_scalar

            # Every other diagonal entry must match exactly.
            elif current_scalar != scalar:
                return {
                    "matches": False,
                    "kernel_type": None,
                    "scalar": None,
                }

    return {
        "matches": True,
        "kernel_type": "identity",
        "delta_power": 0,
        "scalar": scalar,
    }

def delta_burau_matrix(p, n=4):
    """
    Return the Burau matrix of Delta modulo p.

    Delta is the Garside half-twist. In this code it corresponds to the
    longest permutation in S_n, namely `(n-1, ..., 1, 0)`.

    The returned matrix uses the same polynomial-matrix format as
    `burau_mod_p_polynomial_matrix`.
    """
    if p <= 1:
        raise ValueError("p must be >= 2")

    delta_perm = GNF.delta_perm(n)

    # GarsideFactor.artin_factors() returns 0-based adjacent transpositions.
    # The Burau code expects 1-based Artin generators.
    delta_word = [
        idx + 1
        for idx in GarsideFactor(delta_perm).artin_factors()
    ]

    return burau_mod_p_polynomial_matrix(delta_word, p=p, n=n)

def is_projective_delta_matrix(poly_mat, p, n=4):
    """
    Check whether `poly_mat` is a monomial scalar multiple of Burau(Delta).

    Returns a structured result rather than just True/False so the search can
    save the scalar degree and coefficient in its results.
    """
    if p <= 1:
        raise ValueError("p must be >= 2")
    if n < 2:
        raise ValueError("n must be >= 2")

    target = delta_burau_matrix(p=p, n=n)
    size = n - 1
    scalar = None

    def normalize_entry(entry):
        """
        Remove zero coefficients and reduce all coefficients modulo p.

        Polynomial entries should already be reduced, but normalizing here makes
        the comparison robust if a caller builds a matrix by hand.
        """
        normalized = {}
        for exp, coeff in entry.items():
            coeff = coeff % p
            if coeff != 0:
                normalized[exp] = coeff
        return normalized

    for i in range(size):
        for j in range(size):
            target_entry = normalize_entry(target[i][j])
            image_entry = normalize_entry(poly_mat[i][j])

            # Zero entries of Burau(Delta) must stay zero after scalar
            # multiplication.
            if not target_entry:
                if image_entry:
                    return {
                        "matches": False,
                        "kernel_type": None,
                        "delta_power": None,
                        "scalar": None,
                    }
                continue

            # The scalar is a single monomial c*v^k, so the first nonzero
            # target/image pair must both be monomials. This determines c and k.
            if scalar is None:
                if len(target_entry) != 1 or len(image_entry) != 1:
                    return {
                        "matches": False,
                        "kernel_type": None,
                        "delta_power": None,
                        "scalar": None,
                    }

                target_exp, target_coeff = next(iter(target_entry.items()))
                image_exp, image_coeff = next(iter(image_entry.items()))
                if target_coeff % p == 0:
                    return {
                        "matches": False,
                        "kernel_type": None,
                        "delta_power": None,
                        "scalar": None,
                    }

                scalar_exp = image_exp - target_exp
                scalar_coeff = (image_coeff * pow(target_coeff, -1, p)) % p
                scalar = {scalar_exp: scalar_coeff}

            # Once the scalar is known, every entry must equal scalar*target.
            expected_entry = normalize_entry(_poly_mul(scalar, target_entry, p))
            if image_entry != expected_entry:
                return {
                    "matches": False,
                    "kernel_type": None,
                    "delta_power": None,
                    "scalar": None,
                }

    return {
        "matches": scalar is not None,
        "kernel_type": "delta" if scalar is not None else None,
        "delta_power": 1 if scalar is not None else None,
        "scalar": scalar,
    }

def projective_kernel_match(poly_mat, p, n=4):
    """
    Check whether a matrix matches identity or Delta projectively.

    Returns
    -------
    dict
        Example:
        {
            "matches": True,
            "kernel_type": "identity",
            "delta_power": 0,
            "scalar": {3: 5},
        }

        If no match:
        {
            "matches": False,
            "kernel_type": None,
            "delta_power": None,
            "scalar": None,
        }
    """
    identity_match = is_projective_identity_matrix(poly_mat, p=p, n=n)
    if identity_match["matches"]:
        return identity_match

    delta_match = is_projective_delta_matrix(poly_mat, p=p, n=n)
    if delta_match["matches"]:
        return delta_match

    return {
        "matches": False,
        "kernel_type": None,
        "delta_power": None,
        "scalar": None,
    }

def serialize_prefix_state(factor_ids, poly_mat=None, p=None, n=4):
    """
    Convert a prefix into a JSON-friendly dictionary.

    This should be used by MCTS when writing logs. Include enough information
    to reproduce/check the candidate later:
      - factor IDs
      - GNF factors as permutations
      - Artin word
      - Garside length
      - optional projlen
      - optional kernel match
    """
    factor_ids = [int(factor_id) for factor_id in factor_ids]
    factor_perms = factor_ids_to_perms(factor_ids, n=n)

    result = {
        "garside_length": len(factor_ids),
        "factor_ids": factor_ids,
        "gnf_factors": [list(perm) for perm in factor_perms],
        "artin_word": factor_ids_to_artin_word(factor_ids, d=0, n=n),
    }

    if poly_mat is not None:
        min_degree, max_degree = polynomial_matrix_degree_bounds(poly_mat)
        result.update(
            {
                "burau_min_degree": min_degree,
                "burau_max_degree": max_degree,
                "projlen": polynomial_matrix_projlen(poly_mat),
                "support_width": polynomial_matrix_support_width(poly_mat),
            }
        )

        if p is not None:
            result["kernel_match"] = projective_kernel_match(poly_mat, p=p, n=n)

    return result


def run_sanity_checks():
    """
    Run quick checks for the braid helpers that MCTS will depend on.

    These are intentionally small: they catch broken helper wiring without
    doing an expensive search.
    """
    assert len(all_simple_factor_perms(4)) == 24
    assert len(valid_first_factor_ids(4)) == 22

    root = identity_burau_matrix(7)
    assert projective_kernel_match(root, 7)["kernel_type"] == "identity"

    delta = delta_burau_matrix(7)
    assert projective_kernel_match(delta, 7)["kernel_type"] == "delta"

    print("braid_data sanity checks passed")


if __name__ == "__main__":
    run_sanity_checks()
