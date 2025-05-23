import pytest
import copy

from opto import trace
from opto.optimizers import OptoPrime
from opto.utils.llm import LLM


def test_deepcopy_plain_node():
    x = trace.node("x")
    # should not raise
    copy.deepcopy(x)


def test_deepcopy_fun_parameter():
    @trace.bundle(trainable=True)
    def fun(x):
        pass

    # fun.parameter should exist and be deepcopy-able
    copy.deepcopy(fun.parameter)


def test_deepcopy_trainable_node():
    x = trace.node("x", trainable=True)
    # trainable node objects should deep-copy correctly
    copy.deepcopy(x)


def test_deepcopy_optimizer_and_llm():
    # optimizer+LLM may depend on a config file; if it's missing, skip
    x = trace.node("x", trainable=True)
    try:
        optimizer = OptoPrime([x])
        optimizer2 = copy.deepcopy(optimizer)

        llm = LLM()
        copy.deepcopy(llm)
    except FileNotFoundError as e:
        pytest.skip(f"Omit the test: {e}")
