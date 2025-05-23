import pytest
from opto import trace

ops = ['+', '-', '*', '/', '//', '%', '**', '<<', '>>', '&', '|', '^']

@pytest.mark.parametrize("op", ops)
def test_node_binary_ops_against_raw(op):
    x = trace.node(1)
    y = 2

    # x <op> y should equal x.data <op> y
    assert eval(f"x {op} y") == eval(f"x.data {op} y")

    # y <op> x should equal y <op> x.data
    assert eval(f"y {op} x") == eval(f"y {op} x.data")
