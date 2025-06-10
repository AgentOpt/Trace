from opto.trace.nodes import ParameterNode


class Projection:
    """
    Abstract base class for projection methods.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x: ParameterNode) -> ParameterNode:
        """
        Call the projection method on the parameter node `x`.

        Args:
            x: The parameter node to project.

        Returns:
            The projected parameter node.
        """
        return self.project(x)

    def project(self, x: ParameterNode) -> ParameterNode:
        """
        Project the parameter node `x` onto the feasible set.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    