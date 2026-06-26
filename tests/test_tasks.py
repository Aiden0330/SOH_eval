from sohbench.tasks import SOHEstimationTask

def test_task_defines_target_and_metrics():
    t = SOHEstimationTask()
    assert t.target == "soh"
    assert "mae" in t.point_metrics and "mean_bias_error" in t.point_metrics
    assert t.acceptance["generalization"]["mae_max"] > 0
