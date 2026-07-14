import json
import pickle

import numpy as np
import pandas as pd
import pytest

from sohbench.fabric import FeatureState, assert_prefix_invariant
from sohbench.fabric import LegacyStatelessPipeline


def trajectory_table():
    rows = []
    for dataset, cell_id in [("alpha", "a"), ("beta", "b")]:
        for cycle_index in [2, 0, 1]:
            rows.append(
                {
                    "dataset": dataset,
                    "cell_id": cell_id,
                    "cycle_index": cycle_index,
                    "soh": 1.0 - 0.01 * cycle_index,
                    "signal": float(cycle_index),
                }
            )
    return pd.DataFrame(rows)


def test_feature_state_is_immutable_detached_and_serializable():
    source = {"nested": {"values": [1, 2]}, "name": "stable"}

    state = FeatureState.create(source)
    source["nested"]["values"].append(3)

    assert state.payload["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        state.payload["nested"]["values"] += (4,)
    assert FeatureState.from_json(state.to_json()) == state
    assert pickle.loads(pickle.dumps(state)) == state
    assert json.loads(state.to_json())["digest"] == state.digest
    assert state.digest == FeatureState.create({"name": "stable", "nested": {"values": [1, 2]}}).digest


def test_direct_feature_state_construction_also_freezes_the_payload():
    source = {"nested": [1]}
    digest = FeatureState.create(source).digest

    state = FeatureState(payload=source, digest=digest)
    source["nested"].append(2)

    assert state.payload["nested"] == (1,)


def test_feature_state_payload_cannot_be_mutated_through_dict_base_class():
    state = FeatureState.create({"nested": [1]})

    with pytest.raises((AttributeError, TypeError)):
        dict.__setitem__(state.payload, "bypass", 1)


def test_prefix_invariance_rejects_overwriting_an_existing_feature():
    class OverwritingPipeline:
        def fit(self, train_x, train_y, meta):
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            out = frame.copy()
            out["signal"] = out["signal"].max()
            return out

    with pytest.raises(ValueError, match="prefix invariant"):
        assert_prefix_invariant(OverwritingPipeline(), trajectory_table(), pd.DataFrame())


def test_prefix_invariance_keeps_one_fitted_state_for_all_prefixes():
    class CenteredPipeline:
        def fit(self, train_x, train_y, meta):
            return FeatureState.create({"center": float(train_x["signal"].mean())})

        def transform(self, frame, state, meta):
            out = frame.copy()
            out["centered"] = out["signal"] - state.payload["center"]
            return out

    assert_prefix_invariant(CenteredPipeline(), trajectory_table(), pd.DataFrame()) is None


def test_legacy_pipeline_strips_target_and_external_assets_from_meta(tmp_path):
    path = tmp_path / "feature.py"
    path.write_text(
        "def transform(table, meta=None):\n"
        "    assert 'soh' not in table.columns\n"
        "    assert 'soh' not in meta.columns\n"
        "    assert set(meta['cell_id']) == set(table['cell_id'])\n"
        "    out = table.copy()\n"
        "    out['safe'] = 1.0\n"
        "    return out\n",
        encoding="utf-8",
    )
    namespace = {}
    exec(path.read_text(encoding="utf-8"), namespace)
    pipeline = LegacyStatelessPipeline("safe", ["safe"], namespace["transform"], path)
    frame = trajectory_table().iloc[:3].drop(columns=["soh"])
    state = pipeline.fit(frame, np.zeros(len(frame)), pd.DataFrame())
    meta = pd.DataFrame({"dataset": ["alpha", "other"], "cell_id": ["a", "z"], "soh": [1.0, 0.0]})

    transformed = pipeline.transform(frame, state, meta)

    assert "safe" in transformed


def test_prefix_invariance_rejects_future_cycle_normalization():
    class FuturePipeline:
        def fit(self, train_x, train_y, meta):
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            out = frame.copy()
            out["fraction"] = out["cycle_index"] / out["cycle_index"].max()
            return out

    with pytest.raises(ValueError, match="prefix invariant"):
        assert_prefix_invariant(FuturePipeline(), trajectory_table(), pd.DataFrame())


def test_prefix_invariance_accepts_row_local_features_with_unsorted_assets():
    class LocalPipeline:
        def fit(self, train_x, train_y, meta):
            return FeatureState.create({"offset": float(np.mean(train_y))})

        def transform(self, frame, state, meta):
            out = frame.copy()
            out["signal_squared"] = out["signal"] ** 2
            return out

    assert_prefix_invariant(LocalPipeline(), trajectory_table(), pd.DataFrame()) is None
