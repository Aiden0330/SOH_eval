import pytest
from sohbench import registry as R

def test_register_and_get():
    R.clear()
    @R.register_estimator
    class Foo:
        name = "foo"; feature_mode = "unified"; backend = "sklearn"
        def fit(self, X, y, meta): ...
        def predict(self, X, meta): ...
    assert R.get_estimator("foo") is Foo

def test_duplicate_raises():
    R.clear()
    @R.register_estimator
    class A:
        name = "dup"; feature_mode = "unified"; backend = "sklearn"
        def fit(self, X, y, meta): ...
        def predict(self, X, meta): ...
    with pytest.raises(ValueError):
        @R.register_estimator
        class B:
            name = "dup"; feature_mode = "unified"; backend = "sklearn"
            def fit(self, X, y, meta): ...
            def predict(self, X, meta): ...
