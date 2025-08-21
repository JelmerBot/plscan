import builtins
if not hasattr(builtins, "--BUILDING-DOCS--"):
    from .docs import *
else:
    from .docs import *
