from __future__ import annotations
import argparse,json
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import collate_prefixes
from .model import LastFactorTransformer,ModelConfig
from .shards import ShardedPrefixDataset
from .train import run_epoch

def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--batch-size",type=int,default=256);p.add_argument("--device",default="cuda")
    a=p.parse_args();device=torch.device(a.device);ck=torch.load(a.checkpoint,map_location=device,weights_only=False)
    model=LastFactorTransformer(ModelConfig(**ck["model_config"])).to(device);model.load_state_dict(ck["state_dict"])
    results={}
    for split in ("validation","test","extrapolation_test"):
        ds=ShardedPrefixDataset(a.dataset,split);loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,
            collate_fn=partial(collate_prefixes,sparse=bool(ck.get("sparse",True))),num_workers=0,pin_memory=True)
        results[split]=run_epoch(model,loader,device)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=="__main__":main()
