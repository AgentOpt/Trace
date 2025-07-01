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

    # This will raise an error because x and y have different lengths
    # y = [10, 20] 
    # outputs = fun(x, y)
    
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


def test_evaluate(): 
    # This test the evaluate function in opto.trainer.evaluators built on top of batch_run
    from opto.trainer.evaluators import evaluate
    from opto.trainer.guide import AutoGuide
    from opto import trace

    @trace.model
    class MyAgent:
        def __init__(self, param):
            self.param = trace.node(param, trainable=True)            
        
        def forward(self, x):
            y =  x + self.param
            self.param += 1  # This should not affect the batch run
            return y
        
    class MyGuide(AutoGuide):        
        def __init__(self, param):
            super().__init__()
            self.param = param

        def get_feedback(self, query, response, reference=None):
            score = float(response == query + self.param + reference)
            feedback = f"Score: {score}, Response: {response}, Query: {query}"            
            print(score, feedback)
            self.param += 1  # This should not affect the batch run
            return score, feedback
    
    agent = MyAgent(10)
    guide = MyGuide(10)
    inputs = [1, 2, 3, 4, 5]
    infos = [0, 1, 2, 3, 4]  # These are the expected outputs (query + param + info)
    evaluated_scores = evaluate(agent, guide, inputs, infos, num_samples=1, num_threads=1)
    expected_scores = [1, 0, 0, 0, 0]  # All inputs should match the expected outputs
    assert evaluated_scores == expected_scores, f"Expected {expected_scores}, got {evaluated_scores}"   