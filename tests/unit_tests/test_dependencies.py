import pytest
from opto.trace import node, bundle
from opto.trace.utils import contain, sum_feedback

def test_flat_dependencies():
    x = node(1.0, trainable=True)
    y = node(2.0)
    z = x ** y + (x * x * x * x) + 0.5

    assert len(z.parameter_dependencies) == 1
    assert contain(z.parameter_dependencies, x)
    assert not contain(z.parameter_dependencies, y)

def test_nested_dependencies():
    x = node(1.0, trainable=True)
    hidden_param = node(-15.0, trainable=True)

    @bundle()
    def inner_function(x):
        return x ** 2

    @bundle(traceable_code=True)
    def outer_function(x):
        return inner_function(x) + 1 + hidden_param

    output = outer_function(x)

    assert len(output.parameter_dependencies) == 1
    assert contain(output.parameter_dependencies, x)
    assert not contain(output.parameter_dependencies, hidden_param)
    assert len(output.expandable_dependencies) == 1
    assert contain(output.expandable_dependencies, output)

    output.backward('feedback')
    tg = sum_feedback([x])
    tg.visualize()
    sg = tg.expand(output)
    assert len(sg.graph) == 6
    sg.visualize()

def test_hidden_param_only_dependency():
    x = node(1.0)
    hidden_param = node(-15.0, trainable=True)

    @bundle()
    def inner_function(x):
        return x ** 2

    @bundle(traceable_code=True)
    def outer_function(x):
        return inner_function(x) + 1 + hidden_param

    output = outer_function(x)

    assert len(output.parameter_dependencies) == 0
    assert not contain(output.parameter_dependencies, hidden_param)
    assert len(output.expandable_dependencies) == 1
    assert contain(output.expandable_dependencies, output)

    output.backward('feedback')
    tg = sum_feedback([hidden_param])
    tg.visualize()
    tg.expand(output).visualize()

def test_three_layer_hidden_param():
    x = node(1.0)
    hidden_param = node(-15.0, trainable=True)

    @bundle(traceable_code=True)
    def inner_function(x):
        return x ** 2 + hidden_param

    @bundle(traceable_code=True)
    def middle_function(x):
        return inner_function(x) + 1

    @bundle(traceable_code=True)
    def outer_function(x):
        return middle_function(x) + 2

    output = outer_function(x)
    output.backward('test feedback')

    tg = sum_feedback([hidden_param])
    tg.visualize()

    assert len(output.expandable_dependencies) == 1
    x_dep = list(output.expandable_dependencies)[0]
    tg.expand(output).visualize()

    assert len(x_dep.expandable_dependencies) == 1
    y_dep = list(x_dep.info['output'].expandable_dependencies)[0]
    tg.expand(y_dep).visualize()

    assert len(y_dep.expandable_dependencies) == 1
    z_dep = list(y_dep.info['output'].expandable_dependencies)[0]
    tg.expand(z_dep).visualize()
