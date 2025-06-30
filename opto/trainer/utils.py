import asyncio
import functools
import warnings
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm_asyncio
from opto.trace.bundle import ALLOW_EXTERNAL_DEPENDENCIES
from opto.trace.modules import Module

def async_run(runs, args_list = None, kwargs_list = None, max_workers = None, description = None):
    """Run multiple functions in asynchronously.

    Args:
        runs (list): list of functions to run
        args_list (list): list of arguments for each function
        kwargs_list (list): list of keyword arguments for each function
        max_workers (int, optional): maximum number of worker threads to use.
            If None, the default ThreadPoolExecutor behavior is used.
        description (str, optional): description to display in the progress bar.
            This can indicate the current stage (e.g., "Evaluating", "Training", "Optimizing").

    """
    # if ALLOW_EXTERNAL_DEPENDENCIES is not False:
    #     warnings.warn(
    #         "Running async_run with external dependencies check enabled. "
    #         "This may lead to false positive errors. "
    #         "If such error happens, call disable_external_dependencies_check(True) before running async_run.",
    #         UserWarning,
    #     )


    if args_list is None:
        args_list = [[]] * len(runs)
    if kwargs_list is None:
        kwargs_list = [{}] * len(runs)

    async def _run():
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [loop.run_in_executor(executor, functools.partial(run, *args, **kwargs)) 
                    for run, args, kwargs, in zip(runs, args_list, kwargs_list)]
            
            # Use the description in the tqdm progress bar if provided
            if description:
                return await tqdm_asyncio.gather(*tasks, desc=description)
            else:
                return await tqdm_asyncio.gather(*tasks)

    return asyncio.run(_run())


def batch_run(fun, max_workers=None, description=None):
    """
    Create a function that runs in parallel using asyncio, with support for batching.
    The batch size is inferred as the length of the longest argument or keyword argument.            

    Args:
        fun (callable): The function to run.
        
        max_workers (int, optional): Maximum number of worker threads to use.
            If None, the default ThreadPoolExecutor behavior is used.
        description (str, optional): Description to display in the progress bar.

    Returns:
        callable: A new function that processes batches of inputs.

    NOTE: 
        If fun takes input that has __len__ (like lists or arrays), they won't be broadcasted. 
        When using batch_run, be sure to pass list of such arguments of the same length.       

    Example:
        >>> @batch_run(max_workers=4, description="Processing batch")
        >>> def my_function(x, y):
        >>>     return x + y
        >>>     x = [1, 2, 3, 4, 5]
        >>>     y = 10
        >>>     outputs = my_function(x, y)
        >>>     # outputs will be [11, 12, 13, 14, 15]
        >>>     # This will run the function in asynchronously with 4 threads   
    """
    
    
    def _fun(*args, **kwargs):
        
        # We try to infer the batch size from the args
        all_args = args + tuple(kwargs.values())
        # find all list or array-like arguments and use their length as batch size
        batch_size = max(len(arg) for arg in all_args if hasattr(arg, '__len__'))
        
        # broadcast the batch size to all args and record the indices that are broadcasted
        args = [arg if hasattr(arg, '__len__') else [arg] * batch_size for arg in args]
        kwargs = {k: v if hasattr(v, '__len__') else [v] * batch_size for k, v in kwargs.items()}   

        # assert that all args and kwargs have the same length
        lengths = [len(arg) for arg in args] + [len(v) for v in kwargs.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All arguments and keyword arguments must have the same length.")

        # deepcopy if it is a trace.Module (as they may have mutable state)
        # Module.copy() is used to create a new instance with the same parameters
        _args = [arg.copy() if isinstance(arg, Module) else arg for arg in args]
        _kwargs = {k: v.copy() if isinstance(v, Module) else v for k, v in kwargs.items()}

        # Run the forward function in parallel using asyncio with the same parameters. 
        # Since trace.Node is treated as immutable, we can safely use the same instance.
        # The resultant graph will be the same as if we had called the function with the original arguments.

        # convert _args and _kwargs (args, kwargs of list) to lists of args and kwargs

        args_list = [tuple(aa[i] for aa in _args) for i in range(batch_size)]
        kwargs_list = [{k: _kwargs[k][i] for k in _kwargs} for i in range(batch_size)]

        outputs = async_run([fun] * batch_size, args_list=args_list, kwargs_list=kwargs_list,
                            max_workers=max_workers, description=description)
        return outputs

    return _fun


if __name__ == "__main__":

    def tester(t):  # regular time-consuming function
        import time
        print(t)
        time.sleep(t)
        return t, 2

    runs = [tester] * 10  # 10 tasks to demonstrate threading
    args_list = [(3,), (3,), (2,), (3,), (3,), (2,), (2,), (3,), (2,), (3,)]
    kwargs_list = [{}] * 10
    import time
    
    # Example with 1 thread (runs sequentially)
    print("Running with 1 thread (sequential):")
    start = time.time()
    output = async_run(runs, args_list, kwargs_list, max_workers=1)
    print(f"Time with 1 thread: {time.time()-start:.2f} seconds")
    
    # Example with limited workers (2 threads)
    print("\nRunning with 2 threads (parallel):")
    start = time.time()
    output = async_run(runs, args_list, kwargs_list, max_workers=2)
    print(f"Time with 2 threads: {time.time()-start:.2f} seconds")
    
    # Example with limited workers (4 threads)
    print("\nRunning with 4 threads (parallel):")
    start = time.time()
    output = async_run(runs, args_list, kwargs_list, max_workers=4)
    print(f"Time with 4 threads: {time.time()-start:.2f} seconds")
    
    # Example with default number of workers
    print("\nRunning with default number of threads:")
    start = time.time()
    output = async_run(runs, args_list, kwargs_list)
    print(f"Time with default threads: {time.time()-start:.2f} seconds")
