import numpy as np
import copy
import heapq
from dataclasses import dataclass
from typing import Union, List, Tuple, Dict, Any, Optional
from opto import trace
from opto.trace.nodes import ParameterNode
from opto.trainer.utils import async_run, batch_run
from opto.optimizers.utils import print_color
from opto.trainer.algorithms.basic_algorithms import Minibatch, AlgorithmBase, batchify
from opto.trainer.loader import DataLoader
from opto.trainer.sampler import Sampler, RolloutsGraph
import time

# Some helper functions to convert between trace.Module and update_dict

def get_original_name(node):
    """Extract the original name from a node, removing all _copy suffixes."""
    py_name = node.py_name  # This removes colons: "param:0" -> "param0"

    # Find the first occurrence of "_copy" and remove it and everything after
    copy_index = py_name.find('_copy')
    if copy_index != -1:
        return py_name[:copy_index]
    else:
        return py_name

def is_node_copy(a, b):
    """Check if two nodes are copies of each other by comparing their original names.

    This function has transitivity: if A is a copy of B and B is a copy of C,
    then A is also considered a copy of C.
    """
    return get_original_name(a) == get_original_name(b)

def is_module_copy(a, b):
    """ Check if a and b (trace.Modules) are copies of each other. """
    parameters_a = a.parameters() # list of ParameterNode
    parameters_b = b.parameters() # list of ParameterNode
    # Check if all parameters of a are copies of b or vice versa
    # This might over count
    # need to check 1:1 correspondence
    matched = []
    for p_a in parameters_a:
        _matched = []
        for p_b in parameters_b:
            _matched.append(is_node_copy(p_a, p_b))
    np.array(matched)
    if np.all(np.sum(matched, axis=1) == 1) and np.all(np.sum(matched, axis=0) == 1):
        return True
    return False

def remap_update_dict(base_module, update_dict):
    """ Remap the update dict to the agent's parameters. update_dict might have keys which are copies of the base_module's parameters or visa versa.
        This function remaps the keys in update_dict to the original parameters of the base_module.

        The return dict is empty if no keys in update_dict matched any parameters of the base_module. This condition can be used to check if the update_dict contains non-trivial updates.
    """
    parameters = base_module.parameters()  # get the parameters of the base agent
    remapped_update_dict = {}
    for k, v in update_dict.items():
        for p in parameters:
            if is_node_copy(k, p): # Check if k is a copy of p or p is a copy of k
                remapped_update_dict[p] = v
                break # stop checking once we've found a match
    return remapped_update_dict

def set_module_parameters(agent, update_dict):
    """ Set the parameters of the agent based on the update_dict.
        The update_dict is a dictionary of ParameterNode: value pairs.
        The agent's parameters will be updated with the values from the update_dict.
    """
    remapped_update_dict = remap_update_dict(agent, update_dict)  # remap the update dict to the agent's parameters
    for k, v in remapped_update_dict.items():
        k._data = v  # set the parameter's data to the value in the update_dict

def create_module_from_update_dict(agent, update_dict):
    """ Create a new agent from the update_dict.
        The update_dict is a dictionary of ParameterNode: value pairs.
        A new agent will be created with the parameters set to the values from the update_dict.
    """
    new_agent = copy.deepcopy(agent) #.copy()  # create a copy of the agent
    set_module_parameters(new_agent, update_dict)  # set the parameters of the new agent
    return new_agent  # return the new agent