from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from .model_v3 import LastFactorTransformerV3, ModelV3Config


def main():
    parser = argparse.ArgumentParser(description="CUDA smoke test for exact-degree v3")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke test requested, but CUDA is unavailable")
    torch.manual_seed(123)
    config = ModelV3Config(p=5, d_model=96, heads=3, local_layers=1,
                           global_layers=2, ffn_hidden=192, dropout=0.0)
    model = LastFactorTransformerV3(config).to(device)
    coefficients = torch.randint(0, 5, (4, 7, 3, 3), device=device)
    mask = torch.tensor([[1,1,1,1,1,1,1], [1,1,1,1,1,0,0],
                         [1,1,1,0,0,0,0], [1,1,1,1,1,1,0]], dtype=torch.bool, device=device)
    degrees = torch.tensor([[0,1,2,5,9,15,1000000], [0,2,4,8,10000,0,0],
                            [0,500,999999,0,0,0,0], [0,1,7,40,4000,900000,0]], device=device)
    targets = torch.tensor([0, 1, 2, 3], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    logits, descents = model(coefficients, mask, degrees)
    loss = F.cross_entropy(logits.float(), targets) + .1 * descents.float().square().mean()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    if not torch.isfinite(loss): raise RuntimeError("non-finite training loss")
    model.eval()
    with torch.no_grad(): expected = model(coefficients, mask, degrees)[0]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "roundtrip.pt"
        torch.save({"config": config.as_dict(), "state": model.state_dict()}, path)
        saved = torch.load(path, map_location=device, weights_only=False)
        restored = LastFactorTransformerV3(ModelV3Config(**saved["config"])).to(device).eval()
        restored.load_state_dict(saved["state"])
        with torch.no_grad(): actual = restored(coefficients, mask, degrees)[0]
        torch.testing.assert_close(actual, expected)
    result = {
        "status": "clean", "device": str(device), "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__, "parameters": sum(p.numel() for p in model.parameters()),
        "loss": loss.item(), "max_test_degree": int(degrees.max()),
        "forward_backward": "passed", "checkpoint_roundtrip": "passed",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
