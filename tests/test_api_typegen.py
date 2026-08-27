"""Guard: the checked-in frontend types must match the live OpenAPI schema."""

from sohbench.api import typegen

# Endpoints the V1 frontend calls (frontend/app/app.js); losing one of these
# from the schema means the FastAPI surface is no longer a drop-in server.
FRONTEND_OPERATIONS = [
    "GET /api/session",
    "GET /api/catalog",
    "GET /api/profiles",
    "GET /api/examples",
    "GET /api/examples/{example_id}",
    "GET /api/jobs/{job_id}",
    "GET /api/jobs/{job_id}/artifacts",
    "GET /api/inference-jobs/{job_id}",
    "GET /api/inference-jobs/{job_id}/artifacts",
    "GET /api/package/status",
    "GET /api/battery-curves",
    "GET /api/v2/cases/{case_id}",
    "GET /api/v2/cases/{case_id}/review-workspace",
    "GET /api/v2/cases/{case_id}/trajectory",
    "GET /api/v2/prediction-records/{record_id}/curves",
    "POST /api/training/preflight",
    "POST /api/jobs",
    "POST /api/inference-jobs",
    "GET /api/search-jobs",
    "GET /api/search-jobs/{job_id}",
    "POST /api/search-jobs",
    "POST /api/examples/quickstart",
    "POST /api/upload/dataset",
    "POST /api/upload/algorithm",
    "POST /api/upload/featureset",
    "POST /api/upload/model",
    "POST /api/import/battery-soh-plugins",
    "POST /api/sync/native-dashboard",
    "POST /api/package/portable",
    "POST /api/package/standalone",
    "POST /api/v2/cases/resolve",
    "POST /api/v2/cases/{case_id}/identity",
    "POST /api/v2/cases/{case_id}/saved-combinations",
    "POST /api/v2/cases/{case_id}/saved-combinations/activate",
    "POST /api/v2/cases/{case_id}/evidence-selection",
    "POST /api/v2/cases/{case_id}/lifecycle",
    "POST /api/v2/cases/{case_id}/review",
    "POST /api/v2/cases/{case_id}/retests",
    "POST /api/v2/cases/{case_id}/reconcile",
    "POST /api/v2/cases/{case_id}/evidence-packs",
    "POST /api/v2/review-actions/issues",
    "POST /api/v2/review-actions/issues/{issue_id}/status",
    "POST /api/v2/regression-sets",
    "POST /api/v2/regression-sets/{regression_set_id}/members",
    "POST /api/v2/evidence-packs/{pack_id}/sign",
]


def test_generated_frontend_types_are_current():
    for path, content in typegen.generate().items():
        assert path.exists(), (
            f"missing generated file {path}; run `python -m sohbench.api.typegen`"
        )
        assert path.read_text(encoding="utf-8") == content, (
            f"{path} is stale; run `python -m sohbench.api.typegen`"
        )


def test_openapi_covers_every_frontend_operation():
    openapi = typegen.build_openapi()
    operations = {
        f"{method.upper()} {path}"
        for path, item in openapi["paths"].items()
        for method in item
    }
    missing = [op for op in FRONTEND_OPERATIONS if op not in operations]
    assert not missing, f"frontend-called endpoints missing from OpenAPI: {missing}"


def test_operations_map_lists_frontend_endpoints():
    dts = typegen.render_dts(typegen.build_openapi())
    assert '"GET /api/catalog"' in dts
    assert '"POST /api/v2/cases/resolve"' in dts
    assert "export interface ApiOperations" in dts
