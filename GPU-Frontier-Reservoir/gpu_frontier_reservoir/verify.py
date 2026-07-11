from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Exactly verify GPU scalar-identity candidates with peyl")
    p.add_argument("--author-repo", required=True); p.add_argument("--candidates", required=True)
    p.add_argument("--n", type=int, required=True); p.add_argument("--r", type=int, required=True); p.add_argument("--p", type=int, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(a.author_repo).resolve()))
    from peyl.braid import GNF
    from peyl.jonesrep import JonesCellRep
    import numpy as np

    rep = JonesCellRep(n=a.n, r=a.r, p=a.p)
    verified = 0; checked = 0
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with Path(a.candidates).open() as src, out.open("w") as dst:
        for line in src:
            row = json.loads(line); factors = tuple(int(x) for x in row["factors"])
            image = rep.polymat_evaluate_braid(GNF(n=a.n, power=0, factors=factors)) % a.p
            diag = image[0, 0]
            scalar = bool(np.any(diag))
            for i in range(rep.dimension()):
                for j in range(rep.dimension()):
                    scalar &= bool(np.array_equal(image[i, j], diag) if i == j else not np.any(image[i, j]))
            checked += 1; verified += int(scalar)
            dst.write(json.dumps({**row, "exact_scalar_identity": scalar}, sort_keys=True) + "\n")
    print(json.dumps({"checked": checked, "verified": verified, "output": str(out)}, sort_keys=True))


if __name__ == "__main__": main()
