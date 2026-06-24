from vibevalue.evaluate import compute_metrics


def test_compute_metrics_basic():
    preds = [0, 1, 2, 1]
    labels = [0, 1, 2, 2]
    results = compute_metrics(preds, labels)
    assert "accuracy" in results
    assert "precision_0" in results
    assert results["support_0"] == 1
