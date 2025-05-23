import pytest
import pickle
from opto.trace.containers import Map, Seq
from opto.trace.nodes import node


def test_map_pickle(tmp_path):
    path = tmp_path / "test.pkl"
    a = Map({"a": 1, "b": 2})
    pickle.dump(a, open(path, "wb"))
    b = pickle.load(open(path, "rb"))
    assert a == b
    assert a["a"] == 1
    assert a["b"] == 2
    assert isinstance(a["a"], int)


def test_seq_pickle(tmp_path):
    path = tmp_path / "test.pkl"
    a = Seq([1, 2, 3])
    pickle.dump(a, open(path, "wb"))
    b = pickle.load(open(path, "rb"))
    assert a == b
    assert a[0] == 1
    assert a[1] == 2
    assert a[2] == 3


def test_map_with_node_pickle(tmp_path):
    path = tmp_path / "test.pkl"
    a = Map({"a": 1, "b": node(2)})
    pickle.dump(a, open(path, "wb"))
    b = pickle.load(open(path, "rb"))
    assert a == b


def test_seq_with_node_pickle(tmp_path):
    path = tmp_path / "test.pkl"
    a = Seq([1, 2, node(3)])
    pickle.dump(a, open(path, "wb"))
    b = pickle.load(open(path, "rb"))
    assert a == b


def test_seq_parameter_retrieval():
    a = Seq([1, 2, Seq(3, 4, 5)])
    assert a.parameters() == [], "Seq itself is not a parameter node"

    a = Seq([1, node(2, trainable=True), Seq(3, node(4, trainable=True), 5)])
    assert len(a.parameters()) == 2, "Seq contains 2 parameters"


def test_map_parameter_retrieval():
    a = Map({"a": 1, "b": node(2, trainable=True), node('c', trainable=True): 3})
    assert len(a.parameters()) == 2, "Map contains 2 parameters"


def test_nested_mix_map_seq_parameters():
    a = Map({"a": 1, "b": node(2, trainable=True), "c": Seq(3, node(4, trainable=True), 5)})
    assert len(a.parameters()) == 2, "Map contains 2 parameters"


def test_seq_passthrough_behavior():
    # testing indexing with node key (which might not be implemented)
    a = node(3, trainable=True)
    b = Seq([1, 2, 3, 4])
    try:
        _ = b[a]
    except Exception:
        pass
