import pytest

from sohbench.fabric.specs import RunSpec, SpecRef, validate_run_spec_document


def ref(kind: str, name: str) -> SpecRef:
    return SpecRef(kind=kind, spec_id=name, version="1.0.0", digest=("a" * 64))


def run_spec(candidate: str = "ridge") -> RunSpec:
    return RunSpec(
        run_type="training",
        task=ref("task", "soh"),
        dataset=ref("dataset_snapshot", "seven"),
        protocol=ref("protocol", "loco"),
        metric=ref("metric", "soh-core"),
        policy=ref("policy", "fair-v1"),
        candidates=(ref("candidate", candidate),),
        seeds=(0, 1, 2),
    )


def test_candidate_change_does_not_change_comparison_key():
    assert run_spec("ridge").comparison_key == run_spec("gbdt").comparison_key


def test_policy_change_changes_comparison_key():
    first = run_spec()
    second = RunSpec(**{**first.__dict__, "policy": ref("policy", "other")})
    assert first.comparison_key != second.comparison_key


def test_comparison_key_ignores_candidate_run_type_seeds_and_idempotency_key():
    first = run_spec()
    second = RunSpec(
        run_type="inference",
        task=first.task,
        dataset=first.dataset,
        protocol=first.protocol,
        metric=first.metric,
        policy=first.policy,
        candidates=(ref("candidate", "other"),),
        seeds=(99,),
        idempotency_key="request-2",
    )
    assert first.comparison_key == second.comparison_key


def test_to_dict_serializes_tuples_and_is_immutable():
    spec = run_spec()
    document = spec.to_dict()
    assert document["seeds"] == [0, 1, 2]
    assert document["candidates"] == [spec.candidates[0].__dict__]
    with pytest.raises(AttributeError):
        spec.seeds = (4,)


def test_schema_rejects_unknown_run_type():
    document = run_spec().to_dict()
    validate_run_spec_document(document)
    document["run_type"] = "magic"
    with pytest.raises(ValueError, match="run_type"):
        validate_run_spec_document(document)


def test_schema_rejects_unknown_properties_and_bad_digest():
    document = run_spec().to_dict()
    document["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        validate_run_spec_document(document)

    document = run_spec().to_dict()
    document["task"]["digest"] = "A" * 64
    with pytest.raises(ValueError, match="digest"):
        validate_run_spec_document(document)


def test_public_exports_are_available_from_fabric_package():
    from sohbench.fabric import RunSpec as ExportedRunSpec
    from sohbench.fabric import SpecRef as ExportedSpecRef
    from sohbench.fabric import validate_run_spec_document as exported_validator

    assert ExportedRunSpec is RunSpec
    assert ExportedSpecRef is SpecRef
    assert exported_validator is validate_run_spec_document
