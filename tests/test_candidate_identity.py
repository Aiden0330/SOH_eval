"""Training- and inference-derived candidate ids must be byte-identical:
the evidence resolver joins candidates from both derivations on this id,
and a merged candidate pool holds entries from both sources."""

from sohbench.evaluation_service import _artifact_candidate_id
from sohbench.inference import _candidate_id


def test_candidate_id_derivations_agree():
    artifact = (
        "model@sha256:48fc70070ea9a9c4bb64f130d8075584a01c894a38d087e16cb5084d09ec5cb1"
    )
    assert _artifact_candidate_id(artifact) == _candidate_id(artifact)
    assert _artifact_candidate_id("") == _candidate_id("model")
