import warnings
import numpy as np, pandas as pd
from sohkernel.protocols import LeaveOneCellOut, LeaveOneDatasetOut, LeaveOneChemistryOut
from sohkernel.metadata import build_meta

def _table():
    """4-dataset fixture: nasa(LCO), calce_cs2(LCO), hust(LFP), oxford(LCO/NCO)."""
    rows = []
    for ds, cell in [("nasa","c1"),("calce_cs2","c2"),("hust","h1"),("oxford","o1")]:
        for k in range(3):
            rows.append((ds, cell, k, 1.0 - 0.01*k))
    return pd.DataFrame(rows, columns=["dataset","cell_id","cycle_index","soh"])

def test_leave_one_cell_no_leakage():
    t = _table(); meta = build_meta(t)
    for fold in LeaveOneCellOut().generate(t, meta):
        train_cells = set(t.loc[fold.train_idx, "cell_id"])
        test_cells = set(t.loc[fold.test_idx, "cell_id"])
        assert train_cells.isdisjoint(test_cells)           # no cell in both
        assert len(test_cells) == 1

def test_leave_one_dataset_out():
    t = _table(); meta = build_meta(t)
    keys = [f.held_out_key for f in LeaveOneDatasetOut().generate(t, meta)]
    assert set(keys) == {"nasa", "calce_cs2", "hust", "oxford"}

# H5: cross_domain attributes
def test_cross_domain_attrs():
    assert LeaveOneCellOut.cross_domain == False
    assert LeaveOneDatasetOut.cross_domain == True
    assert LeaveOneChemistryOut.cross_domain == True

# H5: per-chemistry folds — 3 folds for 3 chemistries
def test_chemistry_out_three_folds():
    t = _table(); meta = build_meta(t)
    folds = list(LeaveOneChemistryOut().generate(t, meta))
    keys = [f.held_out_key for f in folds]
    assert len(folds) == 3, f"Expected 3 folds, got {len(folds)}: {keys}"

# H5: LCO fold is unflagged (2 datasets: nasa + calce_cs2)
def test_lco_fold_unflagged():
    t = _table(); meta = build_meta(t)
    folds = list(LeaveOneChemistryOut().generate(t, meta))
    lco_keys = [f.held_out_key for f in folds if "LCO" in f.held_out_key]
    assert any(k == "LCO" for k in lco_keys), f"Expected unflagged 'LCO' fold, got: {lco_keys}"

# H5: singleton chemistries are flagged "confounded: 1 dataset"
def test_singleton_chemistries_flagged():
    t = _table(); meta = build_meta(t)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        folds = list(LeaveOneChemistryOut().generate(t, meta))
    keys = [f.held_out_key for f in folds]
    # LFP (hust only) and LCO/NCO (oxford only) must be flagged
    assert any("LFP" in k and "confounded" in k for k in keys), f"LFP should be flagged: {keys}"
    assert any("LCO/NCO" in k and "confounded" in k for k in keys), f"LCO/NCO should be flagged: {keys}"
    # Warnings emitted for singletons
    warn_msgs = [str(warning.message) for warning in w]
    assert any("LFP" in msg or "LCO/NCO" in msg for msg in warn_msgs), f"Expected singleton warnings: {warn_msgs}"

# H5: single-chemistry dataset yields no folds (with warning)
def test_single_chemistry_yields_no_folds():
    t = pd.DataFrame({
        "dataset": ["nasa"]*6,
        "cell_id": ["c1","c1","c1","c2","c2","c2"],
        "cycle_index": [0,1,2,0,1,2],
        "soh": [1.0,0.99,0.98,1.0,0.99,0.98],
    })
    meta = build_meta(t)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        folds = list(LeaveOneChemistryOut().generate(t, meta))
    assert len(folds) == 0
    assert any("one chemistry" in str(warning.message).lower() or
               "single" in str(warning.message).lower()
               for warning in w)


# ---------------------------------------------------------------------------
# forward chaining (Task 7a)
# ---------------------------------------------------------------------------

def _forward_table():
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(11)
    rows = []
    for cell, n in (("a", 50), ("b", 50), ("c", 30), ("short", 10)):
        for k in range(n):
            # f1 tracks age with noise so it is a legitimate feature, not an affine label copy
            rows.append({"dataset": "ds", "cell_id": cell, "cycle_index": n - 1 - k,  # unsorted on purpose
                         "soh": 1.0 - 0.004 * k, "f1": float(k) + rng.normal(0.0, 5.0)})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=3).reset_index(drop=True)


def test_forward_is_registered_and_not_cross_domain():
    from sohkernel.protocols import PROTOCOLS, ForwardChaining
    assert PROTOCOLS["forward"] is ForwardChaining
    assert ForwardChaining().cross_domain is False
    assert ForwardChaining().name == "forward"


def test_forward_holds_out_each_cell_future_and_pools_early_life():
    import warnings
    import numpy as np
    from sohkernel.protocols import ForwardChaining
    t = _forward_table()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        folds = list(ForwardChaining(train_frac=0.6, min_train_cycles=20).generate(t, build_meta(t)))
    assert [f.held_out_key for f in folds] == ["ds/a@forward", "ds/b@forward", "ds/c@forward"]
    assert any("ds/short" in str(w.message) for w in caught)  # too short: warned, never tested
    cutoff = {"ds/a": 30, "ds/b": 30, "ds/c": 20, "ds/short": 10}  # max(20, round(0.6 n)), capped at n
    key = (t["dataset"].astype(str) + "/" + t["cell_id"].astype(str)).to_numpy()
    cyc = t["cycle_index"].to_numpy()
    for f in folds:
        cell = f.held_out_key.removesuffix("@forward")
        assert set(key[f.test_idx]) == {cell}
        assert cyc[f.test_idx].min() > cyc[f.train_idx][key[f.train_idx] == cell].max()
        # every training row is early life: cycle index below its own cell's cutoff
        assert all(cyc[i] < cutoff[key[i]] for i in f.train_idx)
        assert len(f.train_idx) == 30 + 30 + 20 + 10  # pooled early life of all cells
        assert not set(f.train_idx) & set(f.test_idx)
        assert np.array_equal(np.sort(cyc[f.test_idx]), np.arange(cutoff[cell], cutoff[cell] + len(f.test_idx)))


def test_forward_within_cell_mode_trains_only_on_the_held_out_cell():
    from sohkernel.protocols import ForwardChaining
    t = _forward_table()
    key = (t["dataset"].astype(str) + "/" + t["cell_id"].astype(str)).to_numpy()
    for f in ForwardChaining(pooled=False).generate(t, build_meta(t)):
        cell = f.held_out_key.removesuffix("@forward")
        assert set(key[f.train_idx]) == {cell}
        assert len(f.train_idx) == {"ds/a": 30, "ds/b": 30, "ds/c": 20}[cell]


def test_forward_parameters_are_validated():
    import pytest
    from sohkernel.protocols import ForwardChaining
    with pytest.raises(ValueError):
        ForwardChaining(train_frac=1.0)
    with pytest.raises(ValueError):
        ForwardChaining(min_train_cycles=0)


def test_forward_preflight_profile_reports_eligibility():
    from sohkernel import api
    import sohkernel.estimators as E
    E.register_all(include_torch=False)
    t = _forward_table()
    result = api.preflight({"datasets": ["ds"], "estimators": ["ridge"], "featuresets": ["full"],
                            "protocols": ["forward"], "seeds": [0]}, table=t, meta=build_meta(t))
    (combo,) = result.combinations
    assert combo["status"] == "warning" and result.ready
    profile = result.protocols[0]
    assert profile["details"]["eligible_cells"] == 3 and profile["details"]["cells"] == 4
    short = t[t["cell_id"] == "short"]
    blocked = api.preflight({"datasets": ["ds"], "estimators": ["ridge"], "featuresets": ["full"],
                             "protocols": ["forward"], "seeds": [0]}, table=short, meta=build_meta(short))
    assert not blocked.ready and any("forward requires" in r for r in blocked.combinations[0]["reasons"])
    assert api.normalize_request({"protocols": ["forward"]})["protocols"] == ["forward"]


def test_forward_scores_through_the_kernel_api():
    from sohkernel import api
    import sohkernel.estimators as E
    E.register_all(include_torch=False)
    t = _forward_table()
    results = api.score(None, t, build_meta(t), ["forward"], ["ridge"], ["full"], seeds=[0])
    assert set(results["held_out_key"]) == {"ds/a@forward", "ds/b@forward", "ds/c@forward"}
    assert "error" not in results.columns or results["error"].isna().all()
