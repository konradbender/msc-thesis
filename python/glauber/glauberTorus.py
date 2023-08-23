from typing import Any
from .glauberDynIndices import GlauberSimDynIndices
from .glauberFixIndices import GlauberSimulatorFixIndices
from .glauberSim import GlauberSim


class GlauberFixedIndexTorus(GlauberSimulatorFixIndices):

    def __init__(self, *args, **kwargs) -> None:
        kwargs["boundary"] = "random"
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        self.wrap_indices = True

    
class GlauberDynIndexTorus(GlauberSimDynIndices):

    def __init__(self, *args, **kwargs) -> None:
        kwargs["boundary"] = "random"
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        self.wrap_indices = True


    