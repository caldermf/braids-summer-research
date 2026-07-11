import argparse
from pathlib import Path

from .engine import Search, SearchConfig


def main():
    p = argparse.ArgumentParser(description="GPU exhaustive-frontier plus paper-style reservoir search")
    p.add_argument("--table", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--n", type=int, required=True); p.add_argument("--r", type=int, required=True); p.add_argument("--p", type=int, required=True)
    p.add_argument("--frontier-length", type=int, required=True); p.add_argument("--target-length", type=int, required=True)
    p.add_argument("--bucket-size", type=int, default=5000); p.add_argument("--use-best", type=int, default=200000); p.add_argument("--save-best", type=int, default=5000)
    p.add_argument("--degree-window", type=int, required=True); p.add_argument("--boundary-margin", type=int, default=16)
    p.add_argument("--shard-count", type=int, default=1); p.add_argument("--shard-index", type=int, default=0); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--expansion-chunk", type=int, default=50000); p.add_argument("--matmul-chunk", type=int, default=4000); p.add_argument("--device", default="cuda")
    a = p.parse_args()
    cfg = SearchConfig(**{k: v for k, v in vars(a).items() if k not in {"table", "output_dir"}})
    Search(cfg, Path(a.table), Path(a.output_dir)).run()


if __name__ == "__main__": main()
