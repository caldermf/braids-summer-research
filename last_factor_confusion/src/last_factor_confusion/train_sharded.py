from __future__ import annotations

import argparse, json, random
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import collate_prefixes
from .metadata import sha256_file
from .model import LastFactorTransformer, ModelConfig
from .shards import ShardedPrefixDataset, ShardShuffleSampler
from .train import run_epoch


def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--weight-decay",type=float,default=5e-3); p.add_argument("--device",default="cuda")
    p.add_argument("--d-model",type=int,default=256); p.add_argument("--heads",type=int,default=8); p.add_argument("--local-layers",type=int,default=2)
    p.add_argument("--global-layers",type=int,default=4); p.add_argument("--dropout",type=float,default=.08); p.add_argument("--dense",action="store_true")
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    manifest=json.loads((a.dataset/"manifest.json").read_text()); cfg=manifest["config"]
    datasets={s:ShardedPrefixDataset(a.dataset,s) for s in ("train","validation","test","extrapolation_test")}
    collate=partial(collate_prefixes,sparse=not a.dense)
    loaders={s:DataLoader(ds,batch_size=a.batch_size,sampler=ShardShuffleSampler(ds,a.seed) if s=="train" else None,
                         shuffle=False,collate_fn=collate,num_workers=0,pin_memory=True) for s,ds in datasets.items()}
    mc=ModelConfig(p=cfg["prime"],num_classes=22,d_model=a.d_model,heads=a.heads,local_layers=a.local_layers,
                   global_layers=a.global_layers,dropout=a.dropout)
    device=torch.device(a.device); model=LastFactorTransformer(mc).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); a.out_dir.mkdir(parents=True,exist_ok=True)
    best=float("inf"); history=[]
    for epoch in range(1,a.epochs+1):
        row={"epoch":epoch,"train":run_epoch(model,loaders["train"],device,opt),"validation":run_epoch(model,loaders["validation"],device)}
        history.append(row); print(json.dumps(row),flush=True)
        if row["validation"]["cross_entropy"]<best:
            best=row["validation"]["cross_entropy"]
            torch.save({"state_dict":model.state_dict(),"model_config":mc.as_dict(),"sparse":not a.dense,"seed":a.seed,
                        "dataset_manifest":str((a.dataset/"manifest.json").resolve())},a.out_dir/"best_model.pt")
    (a.out_dir/"history.json").write_text(json.dumps(history,indent=2))
    ck=torch.load(a.out_dir/"best_model.pt",map_location=device,weights_only=False); model.load_state_dict(ck["state_dict"])
    results={s:run_epoch(model,loaders[s],device) for s in ("test","extrapolation_test")}
    (a.out_dir/"test_metrics.json").write_text(json.dumps(results,indent=2))
    run={"schema_version":2,"status":"clean","seed":a.seed,"dataset":str(a.dataset.resolve()),"model_config":mc.as_dict(),
         "best_validation_cross_entropy":best,"results":results,"artifact_path":str((a.out_dir/"best_model.pt").resolve()),
         "artifact_checksum":sha256_file(a.out_dir/"best_model.pt")}
    (a.out_dir/"run_manifest.json").write_text(json.dumps(run,indent=2)); print(json.dumps(run,indent=2))


if __name__=="__main__": main()
