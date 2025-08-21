"""
Wraps generated stubs in runtime modules so sphinx can properly document them.

Also applies some type transformations:

- Imports internal classes by their name
- Transforms array-like types to their numpy equivalents

See:
https://github.com/tox-dev/sphinx-autodoc-typehints/issues/161#issuecomment-1398781975
"""

import re
import sys
from textwrap import dedent
from pathlib import Path


def find_definitions_to_mark(stubs):
    defs = []
    for m in re.finditer(r"([ \t]*def )((?!(def|class) ).|\s)*(\"\"\"|\.\.\.)", stubs):
        groups = m.groups()
        if groups[-1] == "...":
            continue  # Skip stubs that are already marked as not implemented
        defs.append((m.start(), m.end(), len(groups[0])))  # start, stop, indent
    return defs


def mark_defs_as_unimplemented(stubs, defs):
    offset = 0
    for _, stop, indent in defs:
        end = stop + offset
        new_line = "\n" + " " * indent + "..."
        stubs = stubs[:end] + new_line + stubs[end:]
        offset += len(new_line)
    return stubs


def find_internal_class_names(stubs):
    return [
        (m.start(2), m.end(2), m.group(2))
        for m in re.finditer(r'(:|->) "(\w+)"', stubs)
    ]


def unquote_class_names(stubs, match_ranges):
    offset = 0
    for start, stop, _ in match_ranges:
        begin = start + offset
        end = stop + offset
        stubs = stubs[: begin - 1] + stubs[begin:end] + stubs[end + 1 :]
        offset -= 2
    return stubs


def internal_class_imports(matches):
    modules = set()
    for _, _, class_name in matches:
        mod_name = re.sub(r"([A-Z])", r"_\1", class_name).lower()
        modules.add((mod_name, class_name))

    return "\n".join(
        f"from .{mod_name} import {class_name}" for mod_name, class_name in modules
    )


def read_stubs(path, stub_name):
    in_file = Path(path) / stub_name
    with open(in_file) as f:
        stubs = f.read()
    return stubs


def write_docs(path, docs_out, stubs):
    docs_file = Path(path) / docs_out
    with open(docs_file, "w") as f:
        f.write(stubs)


def write_wrapper_module(path, ext_name, mod_out, docs_out):
    mods_file = Path(path) / mod_out
    docs_package = docs_out.replace(".py", "")
    with open(mods_file, "w") as f:
        f.write(
            dedent(
                f"""\
                import builtins
                from .{docs_package} import *
                from . import {ext_name}
                __all__ = [name for name in dir({ext_name}) if not name.startswith("_")]
                
                if not hasattr(builtins, "--BUILDING-DOCS--"):
                    for name in dir({ext_name}):
                        globals()[name] = getattr({ext_name}, name)
                else:
                    globals()["__doc__"] = getattr({ext_name}, "__doc__")
                """
            )
        )


def find_array_types(stubs):
    matcher = re.compile(r"Annotated\[ArrayLike, (dict[^:\]]+)\]")
    return [(m.start(), m.end(), eval(m.group(1))) for m in matcher.finditer(stubs)]


def fix_array_types(stubs, array_types):
    offset = 0
    for start, end, flags in array_types:
        num_dims = 1 if flags["shape"] is None else len(flags["shape"])
        dimension = f"tuple[{', '.join(['int'] * num_dims)}]"
        dtype = f"np.dtype[np.{flags['dtype']}]"
        new_annotation = f"np.ndarray[{dimension}, {dtype}]"
        stubs = stubs[: start + offset] + new_annotation + stubs[end + offset :]
        offset -= end - start
        offset += len(new_annotation)
    return stubs


def main(path, stub_name, ext_name, docs_out, mod_out):
    stubs = read_stubs(path, stub_name)

    # Correct .pyi syntax to be valid .py
    defs = find_definitions_to_mark(stubs)
    stubs = mark_defs_as_unimplemented(stubs, defs)

    # Fix other-module-but-in-package class type annotation
    # !! does not work for classes that do not match module name !!
    matches = find_internal_class_names(stubs)
    stubs = unquote_class_names(stubs, matches)
    class_imports = internal_class_imports(matches)
    stubs = stubs.replace("from numpy.typing import ArrayLike", class_imports)
    stubs = stubs.replace("import np", "import numpy as np")

    # Change Annotated Arraylike types
    array_types = find_array_types(stubs)
    stubs = fix_array_types(stubs, array_types)
    stubs = stubs.replace("from typing import Annotated", "import numpy as np")

    write_docs(path, docs_out, stubs)
    write_wrapper_module(path, ext_name, mod_out, docs_out)


if __name__ == "__main__":
    main(*sys.argv[1:6])
