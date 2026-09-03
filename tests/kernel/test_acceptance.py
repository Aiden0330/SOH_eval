from sohkernel.acceptance import evaluate_verdicts
from sohkernel.tasks import SOHEstimationTask

def test_pass_and_fail():
    task = SOHEstimationTask()
    good = {"mae":0.01,"picp":0.92,"monotonicity_violation_rate":0.0,"value_range_violation":0.0}
    bad  = {"mae":0.05,"picp":0.70,"monotonicity_violation_rate":0.10,"value_range_violation":0.2}
    vg = {v.name: v.passed for v in evaluate_verdicts(good, task)}
    vb = {v.name: v.passed for v in evaluate_verdicts(bad, task)}
    assert vg == {"generalization":True,"calibration":True,"physical_consistency":True}
    assert vb == {"generalization":False,"calibration":False,"physical_consistency":False}
