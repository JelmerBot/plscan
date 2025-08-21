"""
Wraps generated stubs in runtime modules so sphinx can properly document them.

See:
https://github.com/tox-dev/sphinx-autodoc-typehints/issues/161#issuecomment-1398781975
"""

import re
import sys
from textwrap import dedent
from pathlib import Path


def find_definitions_to_mark(stubs):
    defs = []
    matcher = re.compile(r"([ \t]*def )((?!(def|class) ).|\s)*(\"\"\"|\.\.\.)")
    for m in matcher.finditer(stubs):
        groups = m.groups()
        if groups[-1] == "...":
            continue  # Skip stubs that are already marked as not implemented
        defs.append(m.start(), m.end(), len(groups[0]))  # start, stop, indent
    return defs


def mark_defs_as_unimplemented(stubs, defs):
    offset = 0
    for _, stop, indent in defs:
        end = stop + offset
        new_line = "\n" + " " * indent + "..."
        stubs = stubs[:end] + new_line + stubs[end:]
        offset += len(new_line)


def find_internal_class_names(stubs):
    matches = []
    for match in re.compile(r": \"(\w+)\",").finditer(stubs):
        matches.append((match.start(1), match.end(1), match.group(1)))
    return matches


def unquote_class_names(stubs, match_ranges):
    offset = 0
    for start, stop, _ in match_ranges:
        begin = start + offset
        end = stop + offset
        stubs = stubs[: begin - 1] + stubs[begin:end] + stubs[end + 1 :]
        offset -= 2


def internal_class_imports(matches):
    modules = set()
    for _, _, class_name in matches:
        mod_name = re.sub(r"([A-Z])", r"_\1", class_name).lower()[1:]
        modules.add((mod_name, class_name))

    return "\n".join(
        f"from .{mod_name} import {class_name}"
        for mod_name, class_name in modules
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


def write_wrapper_module(path, mod_out, docs_out):
    mods_file = Path(path) / mod_out
    docs_package = docs_out.replace(".py", "")
    impl_package = docs_package.replace("_docs", "")
    with open(mods_file, "w") as f:
        f.write(
            dedent(
                f"""\
                import builtins
                if not hasattr(builtins, "--BUILDING-DOCS--"):
                    from .{impl_package} import *
                else:
                    from .{docs_package} import *
                """
            )
        )


def main(path, stub_name, docs_out, mod_out, project_name):
    stubs = read_stubs(path, stub_name)

    # Correct .pyi syntax to be valid .py
    defs = find_definitions_to_mark(stubs)
    stubs = mark_defs_as_unimplemented(stubs, defs)

    # Fix other-module-but-in-package class type annotation
    matches = find_internal_class_names(stubs)
    unquote_class_names(stubs, matches)
    class_imports = internal_class_imports(matches, project_name)
    stubs = stubs.replace("from numpy.typing import ArrayLike", class_imports)

    # Change Annotated Arraylike types
    # stubs = re.sub(r"Annotated\[ArrayLike\]", "numpy.ndarray", stubs)

    
    write_docs(path, docs_out, stubs)
    write_wrapper_module(path, mod_out, docs_out)


if __name__ == "__main__":
    main(*sys.argv[1:6])
