from opto import trace
from opto.optimizers import OptoPrime
from opto.utils.llm import DummyLLM



def test_json_keys():
    """
    Test that the OptoPrimeV2 class correctly initializes with json_keys.
    """
    trace.GRAPH.clear()
    param = trace.node(1, trainable=True)

    def callable(messages,  **kwargs): 
        format_prompt = """Output_format: Your output should be in the following json format, satisfying the json syntax:

{
"reasoning_mod": <Your reasoning>,
"suggestion_mod": {
    <variable_1>: <suggested_value_1>,
    <variable_2>: <suggested_value_2>,
}
}

In "reasoning_mod", explain the problem: 1. what the #Instruction means 2. what the #Feedback on #Output means to #Variables considering how #Variables are used in #Code and other values in #Documentation, #Inputs, #Others. 3. Reasoning about the suggested changes in #Variables (if needed) and the expected result.

If you need to suggest a change in the values of #Variables, write down the suggested values in "suggestion_mod". Remember you can change only the values in #Variables, not others. When <type> of a variable is (code), you should write the new definition in the format of python code without syntax errors, and you should not change the function name or the function signature."""
        assert format_prompt in messages[0]['content']  # system
        assert '"answer":' not in messages[0]['content']
        highlight_prompt = "What are your suggestions on variables int0?"
        assert highlight_prompt in  messages[1]['content']  # user
        return "Dummy response" #messages

    llm = DummyLLM(callable)
    
    optimizer = OptoPrime(
        parameters=[param], 
        llm = llm,       
        json_keys=dict(
            reasoning="reasoning_mod",
            answer=None,
            suggestion="suggestion_mod"),
        highlight_variables=True,
    )
    

    y = param + 10 
    optimizer.zero_feedback()
    optimizer.backward(y, 'dummy feedback')
    optimizer.step(verbose=True)





