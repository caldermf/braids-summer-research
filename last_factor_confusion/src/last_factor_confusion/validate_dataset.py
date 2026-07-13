from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from .metadata import sha256_file
from .shards import atomic_json


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--dataset",type=Path,required=True); a=p.parse_args()
    config=json.loads(a.config.read_text()); manifest={"schema_version":2,"config":config,"status":"clean","splits":{}}
    for split,cfg in config["splits"].items():
        sidecars=sorted((a.dataset/"shards"/split).glob("shard-*.json")); shards=[]; records=trajectories=0
        expected=(cfg["trajectories"]+config["shard_trajectories"]-1)//config["shard_trajectories"]
        if len(sidecars)!=expected: raise SystemExit(f"{split}: expected {expected} shards, found {len(sidecars)}")
        for sidecar in sidecars:
            entry=json.loads(sidecar.read_text()); path=a.dataset/entry["path"]
            if sha256_file(path)!=entry["sha256"]: raise SystemExit(f"checksum mismatch: {path}")
            with np.load(path,allow_pickle=False) as z:
                if len(z["offsets"])!=entry["records"]+1: raise SystemExit(f"bad offsets: {path}")
                if z["coefficients"].shape[1:]!=(3,3): raise SystemExit(f"bad matrix shape: {path}")
            records+=entry["records"]; trajectories+=entry["trajectories"]; shards.append(entry)
        expected_records=cfg["trajectories"]*cfg["prefixes_per_trajectory"]
        if (records,trajectories)!=(expected_records,cfg["trajectories"]): raise SystemExit(f"count mismatch for {split}")
        manifest["splits"][split]={"records":records,"trajectories":trajectories,"shards":shards}
    atomic_json(a.dataset/"manifest.json",manifest); print(json.dumps({k:{x:v[x] for x in ("records","trajectories")} for k,v in manifest["splits"].items()},indent=2))


if __name__=="__main__": main()
