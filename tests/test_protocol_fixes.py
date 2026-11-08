import os, sys, json, numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lggseg import data as D
from lggseg import protocol as P


def test_augmentation_independent_of_worker_copy_and_epoch(monkeypatch):
    ds = D.SegDataset(".", pd.DataFrame({"a": range(8)}), {"mean": [0]*3, "std": [1]*3}, train=True, seed=3)
    draws = lambda ep: [tuple(np.random.default_rng([ds.seed, ep, i]).random(2)) for i in range(8)]
    assert len(set(draws(0))) == 8 and draws(0) != draws(1)
    import copy; c = copy.deepcopy(ds); c.set_epoch(1); assert c.epoch == 1 and ds.epoch == 0


def _mk_run(tmp):
    r = tmp / "run"; r.mkdir()
    (r / "best.pt").write_bytes(b"x"); json.dump({"fitted_on": "train", "mean": [0]*3, "std": [1]*3}, open(r / "channel_stats.json", "w"))
    json.dump({"selected_threshold": 0.5, "config": {"base_channels": 4}, "smoke_test": False}, open(r / "run_meta.json", "w"))
    return r


def test_manifest_tamper_detected(tmp_path):
    r = _mk_run(tmp_path); m = tmp_path / "test_manifest.csv"; m.write_text("patient_id,image_path\nA,a.tif\n")
    out = str(tmp_path / "FROZEN.json"); fr = P.freeze([str(r)], out, pkg_root=str(tmp_path), manifests=[str(m)])
    assert str(m) in fr["manifest_sha256"]; P.verify_frozen(out, pkg_root=str(tmp_path))
    m.write_text("patient_id,image_path\nB,b.tif\n")
    with pytest.raises(P.ProtocolError, match="manifest changed"): P.verify_frozen(out, pkg_root=str(tmp_path))


def test_eval_refuses_freeze_without_test_manifest_hash(tmp_path):
    from test_protocol import _syn_root, _load_ev, make_run
    root = _syn_root(tmp_path); run = make_run(str(tmp_path / "run")); fz = str(tmp_path / "F.json"); P.freeze([run], fz)
    with pytest.raises(SystemExit, match="not hashed"): P.run_once(str(tmp_path / "o"), fz, lambda fr, mk: _load_ev().evaluate(fr, mk, str(root), str(tmp_path / "o")))
