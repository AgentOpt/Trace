import pytest
from opto import trace
bundle = trace.bundle
# Test different decorator usages

def dec(fun):
    # print('dec')
    return fun
def dec2(fun):
    # print('dec')
    return fun

@trace.bundle(\
        )   # random comments
@dec
def my_fun1():  # some comment with bundle
    """ Some def """  # bundle comments
    print('run')  # bundle comments

@bundle()
@dec
def my_fun2():  # some comment with bundle
    """ Some def """  # bundle comments
    print('run')  # bundle comments

@dec2
@bundle()
@dec
def my_fun3():  # some comment with bundle
    """ Some def """  # bundle comments
    print('run')  # bundle comments

@dec2
@trace.bundle()
@dec
def my_fun4():  # some comment with bundle
    """ Some def """  # bundle comments
    print('run')  # bundle comments

def test_bundle_decorator_variants1():
    code_str = '@dec\ndef my_fun1():  # some comment with bundle\n    """ Some def """  # bundle comments\n    print(\'run\')  # bundle comments'
    my_fun1()
    assert my_fun1.info['source'] == code_str, f"EXECPECTED my_fun.info['source'] == code_str\n{my_fun1.info['source']}\n{code_str}"
    assert my_fun1.info['line_number'] == 15

def test_bundle_decorator_variants2():
    code_str = '@dec\ndef my_fun2():  # some comment with bundle\n    """ Some def """  # bundle comments\n    print(\'run\')  # bundle comments'
    my_fun2()
    assert my_fun2.info['source'] == code_str
    assert my_fun2.info['line_number'] == 21

def test_bundle_decorator_variants3():
    code_str = '@dec\ndef my_fun3():  # some comment with bundle\n    """ Some def """  # bundle comments\n    print(\'run\')  # bundle comments'
    my_fun3()
    assert my_fun3.info['source'] == code_str
    assert my_fun3.info['line_number'] == 28

def test_bundle_decorator_variants4():
    code_str = '@dec\ndef my_fun4():  # some comment with bundle\n    """ Some def """  # bundle comments\n    print(\'run\')  # bundle comments'
    my_fun4()
    assert my_fun4.info['source'] == code_str
    assert my_fun4.info['line_number'] == 35
