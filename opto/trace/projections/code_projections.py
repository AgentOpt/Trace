
from opto.trace.projections import Projection
import re
import ast

class BlackCodeFormatter(Projection):
    # This requires the `black` package to be installed.
    
    def project(self, x: str) -> str:
        # importing here to avoid necessary dependencies on black
        # use black formatter for code reformatting
        from black import format_str, FileMode
        if type(x) == str and 'def' in x:
            x = format_str(x, mode=FileMode())
        return x

class DocstringProjection(Projection):
    """
    Projection that formats docstrings.
    """
    def __init__(self, docstring: str):
        self.docstring = docstring    

    def project(self, x: str) -> str:
        """ Replace the docstring in the code wit the stored docstring. """
        if type(x) == str and '"""' in x:
            # replace the docstring in the code with the stored docstring
            x = x.split('"""', 2)
            if len(x) > 2:
                x = f'{x[0]}"""{self.docstring}"""{x[2]}'
            else:
                x = f'{x[0]}"""{self.docstring}"""'
        return x

class SuggestionNormalizationProjection(Projection):
    """
    Normalize LLM-generated suggestion dicts:
      - Literal-eval strings to their true types
      - Alias frequent keys like "__code:8" ↔ "__code8"
      - Black-reformat any code snippets
    """
    def __init__(self, parameters):
        self.parameters = parameters

    def project(self, suggestion: dict) -> dict:
        from black import format_str, FileMode
        def _find_key(node_name: str) -> str | None:
            # exact match
            if node_name in suggestion:
                return node_name
            # strip a colon before digits ("__code:8" → "__code8")
            norm = re.sub(r":(?=\d+$)", "", node_name)
            for k in suggestion:
                if re.sub(r":(?=\d+$)", "", k) == norm:
                    return k
            return None

        normalized: dict = {}
        for node in self.parameters:
            if not getattr(node, "trainable", False):
                continue
            key = _find_key(node.py_name)
            if key is None:
                continue

            raw_val = suggestion[key]
            # re-format any Python defs
            if isinstance(raw_val, str) and "def" in raw_val:
                raw_val = format_str(raw_val, mode=FileMode())

            # convert "123" → 123, "[1,2]" → [1,2], etc.
            target_type = type(node.data)
            if isinstance(raw_val, str) and target_type is not str:
                try:
                    raw_val = target_type(ast.literal_eval(raw_val))
                except Exception:
                    pass

            # map by the parameter’s name, not the node itself
            normalized[node.py_name] = raw_val

        return normalized
