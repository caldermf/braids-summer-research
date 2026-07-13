import json
from pathlib import Path
import numpy as np

from last_factor_confusion.generate_sharded import stratified_lengths
from last_factor_confusion.shards import ShardBucketBatchSampler, ShardedPrefixDataset, write_shard


def test_stratified_lengths_cover_range():
    import random
    values = stratified_lengths(5, 160, 16, random.Random(4))
    assert len(values) == 16
    assert values == sorted(set(values))
    assert 5 <= values[0] and values[-1] <= 160


def test_atomic_ragged_shard(tmp_path: Path):
    records = [
        {"matrix": [[[1,0,0],[0,1,0],[0,0,1]]], "trajectory_id":"a", "prefix_length":5,
         "infimum":0,"projlen":0,"target_class":1,"target_descents":[0]*6},
        {"matrix": [[[1,0,0],[0,1,0],[0,0,1]], [[0,1,0],[0,0,0],[0,0,0]]],
         "trajectory_id":"b","prefix_length":6,"infimum":0,"projlen":1,"target_class":2,"target_descents":[1]*6},
    ]
    path=tmp_path/"x.npz"; entry=write_shard(path,records)
    assert entry["records"]==2 and path.exists()
    with np.load(path,allow_pickle=False) as z:
        assert z["offsets"].tolist()==[0,1,3]
        assert z["coefficients"].shape==(3,3,3)

    manifest = {"splits": {"train": {"shards": [{"path": "x.npz", "records": 2}]}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    dataset = ShardedPrefixDataset(tmp_path, "train")
    batches = list(ShardBucketBatchSampler(dataset, batch_size=1, seed=7))
    assert sorted(index for batch in batches for index in batch) == [0, 1]
