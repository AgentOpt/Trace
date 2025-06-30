from typing import List
from opto import trace
from opto.trainer.utils import batch_run

def test_batch_run_fun():

    @batch_run(max_workers=3)
    def fun(x, y):
        return x + y

    # Create a batch of inputs
    x = [1, 2, 3, 4, 5]
    y = 10   # this will be broadcasted to each element in x

    # Run the function in batch mode
    outputs = fun(x,y)
    assert outputs == [11, 12, 13, 14, 15], f"Expected [11, 12, 13, 14, 15], got {outputs}"

    # Handling a function taking a list as inputs
    @batch_run(max_workers=3)
    def fun(x: List[int], y: List[int]) -> List[int]:
        return [a + b for a, b in zip(x, y)]

    x = [[1, 2, 3], [4, 5, 6]]
    y = [10, 20, 30]  # list won't be braodcasted correctly 

    raise_error = False
    try: 
        outputs = fun(x, y)
    except ValueError as e:
        assert str(e) == "All arguments and keyword arguments must have the same length.", f"Unexpected error: {e}"
        raise_error = True
    assert raise_error, "Expected a ValueError but did not get one."

    # Now we can broadcast y to match the length of x
    y = [[10, 20, 30]] * len(x)  # Broadcast
    outputs = fun(x, y)
    assert outputs == [[11, 22, 33], [14, 25, 36]], f"Expected [[11, 22, 33], [14, 25, 36]], got {outputs}"


    y = [10, 20] # This will raise an error because x and y have different lengths
    raise_error = False
    try:
        outputs = fun(x, y)
    except TypeError as e:
        raise_error = True
    assert raise_error, "Expected a TypeError but did not get one."

def test_batch_run_module():


    @trace.model
    class MyModule:
        def __init__(self, param):
            self.param = trace.node(param, trainable=True)
            self._state = 0
        
        def forward(self, x):
            y =  x + self.param
            self._state += 1  # This should not affect the batch run
            return y
        
    module = MyModule(10)
    x = [1, 2, 3, 4, 5]
    outputs = batch_run(max_workers=3)(module.forward)(x)
    assert outputs == [11, 12, 13, 14, 15], f"Expected [11, 12, 13, 14, 15], got {outputs}"
    param = module.parameters()[0]
    assert len(param.children) == 5


    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50, 60]
    # This should raise an error because x and y have different lengths
    raise_error = False
    try: 
        outputs = batch_run(max_workers=3)(module.forward)(x, y)
    except ValueError as e:
        assert str(e) == "All arguments and keyword arguments must have the same length.", f"Unexpected error: {e}"
        raise_error = True
    assert raise_error, "Expected a ValueError but did not get one."