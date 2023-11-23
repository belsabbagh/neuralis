from src import activations as a


def base_test(activation, x, y, yd):
    fn, der = activation
    assert fn(x) == y
    assert der(x) == yd


def test_identity():
    base_test(a.identity, 2, 2, 1)
