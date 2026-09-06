"""Datasheets (Spec 06): VERIFY detection, chemistry verification, and the lcho preflight warning."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sohkernel import api
from sohkernel.datasheets import chemistry_verified, is_signed, load_datasheets, unverified_fields


def _sheet(name, cathode="LFP", verified_by="VERIFY", signed_by="VERIFY"):
    return {"name": name, "chemistry": {"cathode": cathode, "anode": "graphite", "verified_by": verified_by, "verified_on": "2026-09-03"},
            "cell": {"format": "18650", "rated_capacity_ah": 1.1}, "windows": [{"feature": "q_win_ah", "v_lo": 3.3, "v_hi": 3.5}],
            "signed": {"by": signed_by, "on": "2026-09-03"}}


def test_unverified_fields_walks_nested_structures():
    sheet = _sheet("x", cathode="VERIFY")
    sheet["windows"].append({"feature": "t_win_s", "v_lo": "VERIFY"})
    fields = unverified_fields(sheet)
    assert {"chemistry.cathode", "chemistry.verified_by", "windows[1].v_lo", "signed.by"} <= set(fields)
    assert not chemistry_verified(sheet) and not is_signed(sheet)
    assert chemistry_verified(_sheet("y", verified_by="owner")) and is_signed(_sheet("y", verified_by="owner", signed_by="owner"))


def test_repo_datasheet_drafts_exist_and_nasa_chemistry_is_unverified():
    sheets = load_datasheets(Path(__file__).resolve().parents[2] / "data")
    if not sheets:
        pytest.skip("datasheets not present")
    assert {"nasa", "calce_cs2", "hust", "oxford", "snl_lfp", "uconn", "rwth_isea_public", "snl_nmc", "snl_nca"} <= set(sheets)
    assert not chemistry_verified(sheets["nasa"]) and "chemistry.cathode" in unverified_fields(sheets["nasa"])
    assert sheets["hust"]["windows"][0]["v_lo"] == 3.30 and sheets["hust"]["leak_scan"]["label_source_columns"] == ["charge_capacity_max"]


def test_lcho_preflight_flags_unverified_chemistry(tmp_path):
    import sohkernel.estimators as E
    E.register_all(include_torch=False)
    processed = tmp_path / "processed"
    processed.mkdir()
    sheets = tmp_path / "datasheets"
    sheets.mkdir()
    rng = np.random.default_rng(0)
    for dataset in ("hust", "nasa"):
        rows = []
        for cell in range(3):
            for k in range(70):
                soh = 1.0 - 0.2 * k / 69 + rng.normal(0, 0.003)
                rows.append({"dataset": dataset, "cell_id": f"{dataset}-c{cell}", "cycle_index": k, "soh": soh,
                             "voltage_mean": 3.6 + 0.2 * soh + rng.normal(0, 0.02)})
        pd.DataFrame(rows).to_pickle(processed / f"{dataset}_features.pkl")
    (sheets / "hust.yaml").write_text(yaml.safe_dump(_sheet("hust", verified_by="owner")), encoding="utf-8")
    (sheets / "nasa.yaml").write_text(yaml.safe_dump(_sheet("nasa", cathode="VERIFY")), encoding="utf-8")
    result = api.preflight({"datasets": ["hust", "nasa"], "estimators": ["ridge"], "featuresets": ["full"],
                            "protocols": ["lcho"], "seeds": [0]}, data_dir=tmp_path)
    profile = result.protocols[0]
    assert profile["details"]["unverified_chemistry"] == ["nasa"]
    assert any("cannot support a chemistry claim" in w for w in profile["warnings"])
    assert result.ready and result.combinations[0]["status"] == "warning"


def test_signed_datasheets_agree_with_the_metadata_chemistry_table():
    """D4 (2026-09-03): six datasheets are chemistry-verified and signed; each verified cathode equals the metadata label."""
    from sohkernel.metadata import DATASET_CHEMISTRY, DATASET_C_RATED_AH

    sheets = load_datasheets(Path(__file__).resolve().parents[2] / "data")
    if not sheets:
        pytest.skip("datasheets not present")
    verified = {name for name, sheet in sheets.items() if chemistry_verified(sheet)}
    assert verified == {"calce_cs2", "hust", "oxford", "snl_lfp", "uconn", "rwth_isea_public", "snl_nmc", "snl_nca"}
    for name in verified:
        assert is_signed(sheets[name]), name
        assert sheets[name]["chemistry"]["cathode"] == DATASET_CHEMISTRY[name] == sheets[name]["chemistry"]["metadata_label"], name
    assert not is_signed(sheets["nasa"])
    assert DATASET_CHEMISTRY["rwth_isea_public"] == "NMC" and DATASET_C_RATED_AH["rwth_isea_public"] == 1.85


def test_d12_survey_filled_sampling_provenance_and_licences_from_primary_sources():
    """D12 (2026-09-05): the nine sheets carry measured sampling and raw provenance; only the fields no primary source
    covers stay VERIFY (nasa chemistry, the hust cell model, uconn temperatures); the RWTH licence, profile and temperature
    were closed on 2026-09-06 from Content_RWTH-2021-04545.pdf (CC BY 4.0), leaving only the missing-cell question open."""
    from pathlib import Path

    from sohkernel.datasheets import load_datasheets, unverified_fields

    sheets = load_datasheets(Path(__file__).resolve().parents[2] / "data")
    nine = ["calce_cs2", "hust", "nasa", "oxford", "rwth_isea_public", "snl_lfp", "snl_nca", "snl_nmc", "uconn"]
    assert set(nine) <= set(sheets)
    for name in nine:
        sheet = sheets[name]
        fields = set(unverified_fields(sheet))
        assert "digests.raw" not in fields or name == "rwth_isea_public", name
        assert not any(f.startswith("sampling.") for f in fields), (name, sorted(fields))
        assert not any(f.startswith("windows") for f in fields), (name, sorted(fields))
        assert sheet["provenance"]["raw_files"] or name == "rwth_isea_public", name
        if name in ("hust", "snl_lfp", "snl_nmc", "snl_nca", "uconn", "calce_cs2", "oxford", "nasa"):
            assert isinstance(sheet["sampling"]["rate_hz"], float) and sheet["sampling"]["interval_s"] > 0, name
            assert "source.license" not in fields, name
    assert sheets["hust"]["sampling"]["interval_s"] == 5.0 and sheets["uconn"]["sampling"]["interval_s"] == 1.0
    assert sheets["snl_lfp"]["sampling"]["interval_s"] == 120.024 and sheets["calce_cs2"]["sampling"]["interval_s"] == 30.015
    assert "chemistry.cathode" in unverified_fields(sheets["nasa"])
    rwth = sheets["rwth_isea_public"]
    rwth_open = unverified_fields(rwth)
    assert rwth_open == ["known_issues[0]"], rwth_open  # only the missing-cell question is left
    assert rwth["source"]["license"].startswith("CC BY 4.0") and rwth["protocol"]["temperature_c"] == [25]
    assert rwth["protocol"]["segments"][0]["rate"].startswith("Content.pdf §3: one cycle = 30 min discharge to 3.5 V")
    assert sheets["rwth_isea_public"]["sampling"]["rate_hz"] == "not applicable"
