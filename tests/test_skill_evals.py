"""L2: every skill in this repository ships at least three eval cases (prompt, expects, must_not) beside its SKILL.md."""
from pathlib import Path

import yaml

SKILLS = Path(__file__).resolve().parents[1] / ".claude" / "skills"


def test_every_skill_has_three_well_formed_eval_cases():
    skills = sorted(p.parent for p in SKILLS.glob("*/SKILL.md"))
    assert len(skills) >= 4
    for skill in skills:
        payload = yaml.safe_load((skill / "evals.yaml").read_text(encoding="utf-8"))
        assert payload["skill"] == skill.name
        cases = payload["cases"]
        assert len(cases) >= 3, skill.name
        for case in cases:
            assert case["id"].startswith(skill.name) and case["prompt"].strip()
            assert isinstance(case["expects"], list) and case["expects"], case["id"]
            assert isinstance(case.get("must_not", []), list)
