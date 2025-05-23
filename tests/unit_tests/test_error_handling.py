import os
import pytest
from opto.trace.bundle import bundle, ExecutionError
from opto.trace.nodes import Node, node, ExceptionNode
from opto.trace import model
from opto.optimizers.optoprime import OptoPrime

x = Node(1, name="node_x")
y = Node(0, name="node_y")


def test_division_by_zero_in_program():
    def bug_program(x: Node, y: Node):
        return x / y

    with pytest.raises(ExecutionError) as e:
        bug_program(x, y)
    print(f"Error message to developer:\n{e.value}")
    print(f"Error message to optimizer:\n{e.value.exception_node.data}")
    assert isinstance(e.value.exception_node, ExceptionNode)
    assert x in e.value.exception_node.parents
    assert y in e.value.exception_node.parents


def test_decorator_error_fun():
    @bundle()
    def error_fun():
        x = None
        x.append(1)

    with pytest.raises(ExecutionError) as e:
        error_fun()
    print(f"Error message to developer:\n{e.value}")
    print(f"Error message to optimizer:\n{e.value.exception_node.data}")


def test_inline_error_fun():
    def error_fun():
        x = None
        x.append(1)

    error_fun = bundle()(error_fun)
    with pytest.raises(ExecutionError) as e:
        error_fun()
    print(f"Error message to developer:\n{e.value}")
    print(f"Error message to optimizer:\n{e.value.exception_node.data}")


def test_nested_error():
    def error_fun():
        x = None
        x.append(1)

    @bundle()
    def top_fun(x):
        x += 1
        error_fun()
        return 2

    with pytest.raises(ExecutionError) as e:
        top_fun(1)
    print(f"Error message to developer:\n{e.value}")
    print(f"Error message to optimizer:\n{e.value.exception_node.data}")


def test_syntax_error_in_trainable_code():
    code = """
def bug_progam(x):
    x = 1
    x *=2
    x . 10 # syntax error
    return
"""
    @bundle(trainable=True)
    def bug_progam(x):
        x + 10

    bug_progam.parameter._data = code
    with pytest.raises(ExecutionError) as e:
        bug_progam(1)
    print(f"Error message to developer:\n{e.value}")
    print(f"Error message to optimizer:\n{e.value.exception_node.data}")
    assert isinstance(e.value.exception_node, ExceptionNode)
    assert bug_progam.parameter in e.value.exception_node.parents
    assert "SyntaxError" in e.value.exception_node.data


def test_execution_error_in_trainable_code():
    @bundle(trainable=True)
    def bug_progam(x):
        x + 10
        x / 0

    with pytest.raises(ExecutionError) as e:
        bug_progam(1)
    print(f"Error message to developer:\n{e.value}")
    print(f"Error message to optimizer:\n{e.value.exception_node.data}")
    assert bug_progam.parameter in e.value.exception_node.parents


def test_nested_execution_error_in_trainable_code():
    def bug_progam(x):
        x + 10
        x / 0

    @bundle(trainable=True)
    def top_fun(x):
        bug_progam(x)

    with pytest.raises(ExecutionError) as e:
        top_fun(1)
    print(f"Error message to developer:\n{e.value}")
    print(f"Error message to optimizer:\n{e.value.exception_node.data}")
    assert top_fun.parameter in e.value.exception_node.parents


def test_error_in_comprehension_scope():
    @bundle(trainable=True)
    def top_fun(x):
        if False:
            u = [1]
        x = [u[i] for i in range(3)]

    with pytest.raises(ExecutionError) as e:
        top_fun(1)
    print(f"Error message to developer:\n{e.value}")
    print(f"Error message to optimizer:\n{e.value.exception_node.data}")
    assert top_fun.parameter in e.value.exception_node.parents


def test_unpack_none_error():
    @bundle(catch_execution_error=True)
    def fun(x):
        return None

    with pytest.raises(ExecutionError) as e:
        a, b = fun(1)
    print(f"Error message to developer:\n{e.value}")
    assert isinstance(e.value.exception_node, ExceptionNode)


def test_lambda_capture_error():
    @bundle()
    def test(a, b):
        return a(b)

    def add_one(y):
        add_one_fn = lambda x: x + y + 1
        return add_one_fn

    add_one_fn = add_one(2)
    with pytest.raises(ExecutionError) as e:
        test(add_one_fn, '1')
    print(f"Error message to developer:\n{e.value}")
    print(f"Error message to optimizer:\n{e.value.exception_node.data}")
    assert isinstance(e.value.exception_node, ExceptionNode)


def test_early_exception():
    @model
    class TestAgent:
        @bundle(trainable=True)
        def func1(self):
            return 1

        @bundle(trainable=True)
        def func2(self):
            return 1

        @bundle(trainable=True)
        def func3(self):
            raise Exception("Error in func1")

        def act(self):
            self.func1()
            self.func2()
            self.func3()

    agent = TestAgent()
    with pytest.raises(ExecutionError) as e:
        output = agent.act()

    feedback = e.value.exception_node.create_feedback()
    output = e.value.exception_node
    optimizer = OptoPrime(agent.parameters())
    optimizer.zero_feedback()
    optimizer.backward(output, feedback)
    optimizer.summarize()
