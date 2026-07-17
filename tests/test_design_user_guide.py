from pathlib import Path

from scripts.validate_design_user_guide import REQUIRED_SECTIONS, validate_document


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "soh-evaluation-system-design-and-user-guide.html"


def test_design_user_guide_satisfies_offline_structure_contract():
    issues = validate_document(GUIDE, source_root=ROOT)
    assert issues == []


def test_design_user_guide_contains_every_required_chapter():
    text = GUIDE.read_text(encoding="utf-8")
    for section_id in REQUIRED_SECTIONS:
        assert f'id="{section_id}"' in text


def test_design_user_guide_rejects_multitoken_stylesheet_rel(tmp_path):
    guide = tmp_path / "guide.html"
    guide.write_text(
        GUIDE.read_text(encoding="utf-8").replace(
            "</head>",
            '<link rel="alternate stylesheet" href="guide.css">\n</head>',
        ),
        encoding="utf-8",
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "external-asset" for issue in issues)
