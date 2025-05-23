import copy
import numpy as np
from opto.trace import node
from opto.trace import operators as ops
from opto.trace.utils import contain


def test_add_node_str():
    x = node("NodeX")
    y = node("NodeY")
    z = ops.add(x=x, y=y)
    assert z.data == x.data + y.data
    assert x in z.parents and y in z.parents
    assert z in x.children and z in y.children
    for k, v in z._inputs.items():
        assert locals()[k] == v


def test_join_node_str():
    x = node("NodeX")
    y = node("NodeY")
    z = node('+').join([x, y])
    assert z.data == x.data + '+' + y.data


def test_add_node_int():
    x = node(1)
    y = node(2)
    z = ops.add(x, y)
    assert z.data == x.data + y.data
    assert x in z.parents and y in z.parents
    assert z in x.children and z in y.children
    for k, v in z._inputs.items():
        assert locals()[k] == v


def test_conditional_operator():
    x = node(1)
    y = node(2)
    condition = node(True)
    z = ops.cond(condition, x, y)
    assert z.data == x.data if condition.data else y.data
    assert x in z.parents and y in z.parents and condition in z.parents
    assert z in x.children and z in y.children and z in condition.children
    for k, v in z._inputs.items():
        assert locals()[k] == v


def test_getitem_list_of_nodes():
    index = node(0)
    x = node([node(1), node(2), node(3)])
    z = ops.getitem(x, index)
    assert z == x[index]
    assert z is not x[index]
    assert z.data == x.data[index.data].data
    assert x in z.parents and index in z.parents
    assert z in x.children and z in index.children
    for k, v in z._inputs.items():
        assert locals()[k] == v


def test_getitem_list():
    index = node(0)
    x = node([1, 2, 3])
    z = ops.getitem(x, index)
    assert z == x[index]
    assert z.data == x.data[index.data]
    assert x in z.parents and index in z.parents
    assert z in x.children and z in index.children
    for k, v in z._inputs.items():
        assert locals()[k] == v


def test_iterables_nodes_and_dict():
    x = node([1, 2, 3])
    for k, v in enumerate(x):
        assert v.data == x.data[k]

    x = node(dict(a=1, b=2, c=3))
    for k, v in x.items():
        assert v.data == x.data[k.data]


def test_node_copy_clone_deepcopy():
    x = node([1, 2, 3])
    z = ops.getitem(x, node(0))
    z_new = ops.identity(z)
    z_clone = z.clone()
    z_copy = copy.deepcopy(z)
    assert z_new.data == z.data
    assert z_clone.data == z.data
    assert z_copy.data == z.data
    assert contain(z_new.parents, z) and len(z_new.parents) == 1 and contain(z.children, z_new)
    assert contain(z_clone.parents, z) and len(z_clone.parents) == 1 and contain(z.children, z_clone)
    assert not contain(z_copy.parents, z) and len(z_copy.parents) == 0 and not contain(z.children, z_copy)


def test_magic_function_operator():
    x = node("NodeX")
    y = node("NodeY")
    z = x + y
    assert z.data == x.data + y.data
    assert x in z.parents and y in z.parents
    assert z in x.children and z in y.children
    for k, v in z._inputs.items():
        assert locals()[k] == v


def test_boolean_operators():
    x = node(1)
    y = node(2)
    z = x < y
    assert z.data == x.data < y.data
    assert bool(z) is True


def test_hash_and_equality():
    x = node(1)
    y = node(1)
    assert y in [x]
    assert y not in {x}
    assert hash(x) != hash(y)


def test_callable_node():
    def fun(x):
        return x + 1

    fun_node = node(fun)
    output = fun_node(node(2))
    assert output == 3
    assert len(output.parents) == 2


def test_trainable_wrapping():
    a = []
    x = node(a, trainable=True)
    y = node(x, trainable=True)
    assert x.data is y.data

    x = node(a, trainable=False)
    y = node(x, trainable=True)
    assert x.data is y.data


def test_node_description():
    x = node(1, description="x")
    assert x.description == "[Node] x"

    y = node(1)
    assert y.description == '[Node] This is a node in a computational graph.'

    x = node(1, description="x", trainable=True)
    assert x.description == "[ParameterNode] x"

    x = node(1, trainable=True)
    assert x.description == "[ParameterNode] This is a ParameterNode in a computational graph."


def test_iterating_numpy_array():
    x = node(np.array([1, 2, 3]))
    for i, v in enumerate(x):
        assert isinstance(v, type(x))
        assert v.data == x.data[i]
