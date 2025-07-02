from opto.trace.modules import Module, model
from opto.trace.nodes import node
from opto.trace.bundle import bundle
import os
import pickle

# Test Module as a class


class BaseModule(Module):
    def __init__(self):
        super().__init__()
        self._param = node(1, trainable=True)

    @bundle(trainable=True)
    def method1(self, x):
        return x

    def method2(self, y):
        return y

    def forward(self, i):
        return self.method1(i)

base = BaseModule()
assert len(base.parameters()) == 2
assert len(base.parameters_dict()) == 2


def dummy_method():
    return 1

# test inheritance
class ChildModule(BaseModule):
    def __init__(self):
        super().__init__()
        self._extra_param = node(1, trainable=True)
        self._extra_method = bundle(trainable=True)(dummy_method)
        self._base = BaseModule()  # ParameterContainer

    @bundle(trainable=True)
    def method1(self, x):
        return x

    def method2(self, y):
        return y

child = ChildModule()
print(child.parameters_dict().keys())
assert len(child.parameters()) == 6
assert len(child.parameters_dict()) == 5


# Test using model decorator
@model
class BaseClass:
    def __init__(self):
        super().__init__()
        self._param = node(1, trainable=True)

    @bundle(trainable=True)
    def method1(self, x):
        return x

    def method2(self, y):
        return y

    def forward(self, i):
        return self.method1(i)


def test_model_decorator():
    base = BaseClass()
    assert len(base.parameters()) == 2
    assert len(base.parameters_dict()) == 2


def dummy_method():
    return 1

# test inheritance
class ChildClass(BaseClass):
    def __init__(self):
        super().__init__()
        self._extra_param = node(1, trainable=True)
        self._extra_method = bundle(trainable=True)(dummy_method)
        self._base = BaseClass()  # ParameterContainer

    @bundle(trainable=True)
    def method1(self, x):
        return x

    def method2(self, y):
        return y

def test_inheritance():
    child = ChildClass()
    assert len(child.parameters()) == 6, f"Expected 6 parameters, got {child.parameters_dict()}"
    assert len(child.parameters_dict()) == 5


# test save and load
def test_save_load_pickle():
    child = ChildClass()
    child._extra_param._data = 2  # simulate data changes
    child._extra_method.parameter._data = "fake method" # simulate data changes
    child._base._param._data = 3  # simulate data changes
    child._new_param = node(1, trainable=True)  # simulate adding new parameter
    assert len(child.parameters()) == 7

    try:
        child.save("test.pkl")
    except AttributeError:
        print("Cannot save attributes of classes created by @model decorator")
        pass

    child._base = BaseModule()  # can save Modules
    child._base._param._data = 3  # simulate data changes
    try:
        child.save("test.pkl")
    except AttributeError:
        print("Cannot save classes created by @model decorator")

# child2 = ChildClass()
# child2.load("test.pkl")
# os.remove("test.pkl")

# assert child2._extra_param == 2
# assert child2._extra_method.parameter._data == "fake method"
# assert child2._base._param._data == 3
# assert child2._new_param == 1 # simulate new parameter

# Test case: testing multiple inheritance
class NonModuleBaseClass():
    def __init__(self):
        pass

    @bundle()
    def method1(self):
        return 1

@model
class ChildClass2(NonModuleBaseClass):
    def __init__(self):
        super().__init__()

    @bundle(trainable=True)
    def method2(self, x):
        return self.method1() + x

    def forward(self, i):
        return self.method2(i)

def test_multiple_inheritance():
    child = ChildClass2()
    result = child.forward(1)
    assert result._data == 2


# Test cases for model_dump
@model
class DummyClass:
    def __init__(self):
        super().__init__()
        self._param = node(1, trainable=True)
        self.regular_attr = "test"

    @bundle(trainable=True)
    def regular_method(self, x):
        return x

    def __str__(self):
        return "DummyClass"

    def __custom__(self):
        return "custom"

@model
class ComplexClass:
    def __init__(self):
        super().__init__()
        self._param = node(1, trainable=True)
        self._nested = DummyClass()

    @bundle(trainable=True)
    def complex_method(self, x):
        return self._nested.regular_method(x)

    def __str__(self):
        return "ComplexClass"

def test_model_dump_basic():
    dummy = DummyClass()
    dummy._param._data = 42  # Change the node value
    temp_file = "temp_dummy.py"
    try:
        dummy.model_dump(temp_file)
        with open(temp_file, "r") as f:
            content = f.read()
            # Check if class definition is present
            assert "class DummyClass:" in content
            # Check if regular method is present
            assert "def regular_method" in content
            # Check if __str__ is present (overridden dunder)
            assert "def __str__" in content
            # Check if __custom__ is present (custom dunder)
            assert "def __custom__" in content
            # Check if regular attribute is present
            assert "regular_attr" in content
            # Check if node initialization was replaced with current value
            assert "self._param = 42" in content
            assert "self._param = node(1" not in content
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def test_model_dump_complex():
    complex_obj = ComplexClass()
    temp_file = "temp_complex.py"
    try:
        complex_obj.model_dump(temp_file)
        with open(temp_file, "r") as f:
            content = f.read()
            # Check if class definition is present
            assert "class ComplexClass:" in content
            # Check if complex method is present
            assert "def complex_method" in content
            # Check if __str__ is present
            assert "def __str__" in content
            # Check if nested class reference is in the method
            assert "self._nested.regular_method" in content
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def test_model_dump_with_projection():
    dummy = DummyClass()
    temp_file = "temp_dummy_formatted.py"
    try:
        # Test with BlackCodeFormatter
        from opto.trace.projections import BlackCodeFormatter
        dummy.model_dump(temp_file, projections=[BlackCodeFormatter()])
        with open(temp_file, "r") as f:
            content = f.read()
            # Check if content is properly formatted
            assert "class DummyClass:" in content
            assert "def regular_method" in content
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@model
class NonTrainableClass:
    def __init__(self):
        super().__init__()
        self._param = node(1, trainable=False)
        self._param2 = node(2, trainable=False)
        self.regular_attr = "test"

    @bundle(trainable=False)
    def non_trainable_method(self, x):
        return x

    @bundle(trainable=False)
    def another_non_trainable(self, y):
        return y + 1

def test_model_dump_non_trainable():
    obj = NonTrainableClass()
    obj._param._data = 10  # Change node value
    obj._param2._data = 20  # Change another node value
    temp_file = "temp_non_trainable.py"
    try:
        obj.model_dump(temp_file)
        with open(temp_file, "r") as f:
            content = f.read()
            # Check if class definition is present
            assert "class NonTrainableClass:" in content
            # Check if node initializations were replaced with current values
            assert "self._param = 10" in content
            assert "self._param2 = 20" in content
            # Verify no node() calls remain
            assert "node(" not in content
            # Verify no bundle decorators remain
            assert "@bundle" not in content
            # Check if methods are present but without decorators
            assert "def non_trainable_method" in content
            assert "def another_non_trainable" in content
            # Check if regular attribute is present
            assert "regular_attr" in content
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def test_model_dump_mixed_trainable():

    @model
    class MixedClass:
        def __init__(self):
            super().__init__()
            self._trainable = node(1, trainable=True)
            self._non_trainable = node(2, trainable=False)
            self.regular_attr = "test"

        @bundle(trainable=True)
        def trainable_method(self, x):
            return x

        @bundle(trainable=False)
        def non_trainable_method(self, y):
            return y + 1


    obj = MixedClass()
    obj._trainable._data = 100
    obj._non_trainable._data = 200

    obj.trainable_method.parameter._data = "def trainable_method(self, x):\n     return x + 3"

    temp_file = "temp_mixed.py"
    try:
        obj.model_dump(temp_file)
        with open(temp_file, "r") as f:
            content = f.read()
            # Check if class definition is present
            assert "class MixedClass:" in content
            # Check if all node initializations were replaced
            assert "self._trainable = 100" in content
            assert "self._non_trainable = 200" in content
            # Verify no node() calls remain
            assert "node(" not in content
            # Verify no bundle decorators remain
            assert "@bundle" not in content
            # Check if methods are present but without decorators
            assert "def trainable_method" in content
            assert "return x + 3" in content
            assert "def non_trainable_method" in content
            # Check if regular attribute is present
            assert "regular_attr" in content
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def test_model_dump_and_import():
    @model
    class StrangeCalculator:
        def __init__(self):
            super().__init__()
            self.offset = node(2, trainable=True)
            self.multiplier = node(1.5, trainable=True)
        
        @bundle(trainable=True)
        def add(self, x, y):
            """Add two numbers with an offset"""
            return x + y + self.offset
        
        @bundle(trainable=True)
        def multiply(self, x, y):
            """Multiply two numbers with a multiplier"""
            return x * y * self.multiplier
    
    # Create instance and modify parameters
    calc = StrangeCalculator()
    calc.offset._data = 3
    calc.multiplier._data = 2.0
    calc.add.parameter._data = "def add(self, x, y):\n    return x + y + self.offset + 1"
    calc.multiply.parameter._data = "def multiply(self, x, y):\n    return x * y * self.multiplier * 2"
    
    # Dump the model
    temp_file = "temp_calculator.py"
    try:
        calc.model_dump(temp_file)
        
        # Import the dumped class
        import importlib.util
        spec = importlib.util.spec_from_file_location("temp_calculator", temp_file)
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)
        
        # Get the imported class
        ImportedCalculator = temp_module.StrangeCalculator
        
        # Create instance and test functionality
        imported_calc = ImportedCalculator()
        
        # Test the modified behavior
        result_add = imported_calc.add(5, 3)
        result_multiply = imported_calc.multiply(4, 2)
        
        # Verify the results match our expected modified behavior
        # add: 5 + 3 + 3 + 1 = 12
        # multiply: 4 * 2 * 2.0 * 2 = 32
        assert result_add == 12, f"Expected 12, got {result_add}"
        assert result_multiply == 32, f"Expected 32, got {result_multiply}"
        
        # Verify the attributes have the correct values
        assert imported_calc.offset == 3
        assert imported_calc.multiplier == 2.0
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def test_copy_function():
    """Test the copy function of Module class."""
    
    @model
    class TestCopyClass:
        def __init__(self):
            super().__init__()
            self._param = node(10, trainable=True)
            self.regular_attr = "original_value"
            self.list_attr = [1, 2, 3]
            self.dict_attr = {"key": "value"}
        
        @bundle(trainable=True)
        def test_method(self, x):
            return x + self._param
        
        def forward(self, x):
            return self.test_method(x)
    
    # Create original instance
    original = TestCopyClass()
    original.regular_attr = "modified_value"
    original.list_attr.append(4)
    original.dict_attr["new_key"] = "new_value"
    
    # Create a copy
    copied = original.copy()
    
    # Test that it's a different object
    assert copied is not original
    
    # Test that regular attributes are copied (deep copy)
    assert copied.regular_attr == "modified_value"
    assert copied.list_attr == [1, 2, 3, 4]
    assert copied.dict_attr == {"key": "value", "new_key": "new_value"}
    
    # Test that parameters are references to the original parameters
    assert copied._param is original._param
    assert copied.test_method.parameter is original.test_method.parameter
    
    # Test that modifying the original parameter affects the copy
    original._param._data = 20
    assert copied._param._data == 20
    
    # Test that modifying the copy's parameter affects the original
    copied._param._data = 30
    assert original._param._data == 30
    
    # Test that the copy can still function
    result = copied.forward(5)
    assert result._data == 35  # 5 + 30
    
    # Test that modifying regular attributes doesn't affect the original
    copied.regular_attr = "copy_only_value"
    assert original.regular_attr == "modified_value"
    
    # Test that modifying list/dict attributes doesn't affect the original (deep copy)
    copied.list_attr.append(5)
    assert len(original.list_attr) == 4
    assert len(copied.list_attr) == 5
    
    copied.dict_attr["copy_only"] = "copy_value"
    assert "copy_only" not in original.dict_attr
    assert "copy_only" in copied.dict_attr

def test_copy_function_with_nested_modules():
    """Test the copy function with nested modules."""
    
    @model
    class NestedModule:
        def __init__(self):
            super().__init__()
            self._nested_param = node(5, trainable=True)
        
        @bundle(trainable=True)
        def nested_method(self, x):
            return x * self._nested_param
        
        def forward(self, x):
            return self.nested_method(x)
    
    @model
    class ParentModule:
        def __init__(self):
            super().__init__()
            self._param = node(10, trainable=True)
            self._nested = NestedModule()
            self.regular_attr = "parent_value"
        
        @bundle(trainable=True)
        def parent_method(self, x):
            return self._nested.forward(x) + self._param
        
        def forward(self, x):
            return self.parent_method(x)
    
    # Create original instance
    original = ParentModule()
    original.regular_attr = "modified_parent"
    original._nested._nested_param._data = 7
    
    # Create a copy
    copied = ParentModule()
    copied = original.copy()
    
    # Test that it's a different object
    assert copied is not original
    
    # Test that nested module is copied but parameters are references
    assert copied._nested is not original._nested  # Different object
    assert copied._nested._nested_param is original._nested._nested_param  # Same parameter reference
    
    # Test that regular attributes are copied
    assert copied.regular_attr == "modified_parent"
    
    # Test that modifying nested parameter affects both
    original._nested._nested_param._data = 8
    assert copied._nested._nested_param._data == 8
    
    # Test that the copy can still function
    result = copied.forward(3)
    assert result._data == 34  # (3 * 8) + 10
