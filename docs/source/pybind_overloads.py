"""Sphinx extension documenting each overload of a pybind11 function as a separate entry.

pybind11 joins the docstrings of all overloads into a single ``__doc__``, which autodoc would then render
as a single entry. Each overload's docstring is expected to start with its signature, which is used here to
split ``__doc__`` back into one entry per overload.

Used in place of ``automethod`` (nested in an ``autoclass``, with the method in ``:exclude-members:``) or
``autofunction``:

.. code-block:: rst

    .. autooverloads:: pyehm.utils.gen_clusters
"""
import inspect
import pydoc
import re

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.ext.napoleon import NumpyDocstring
from sphinx.util.docutils import SphinxDirective


def split_overloads(name, doc):
    """Split ``doc`` into a list of ``(signature, body_lines)`` tuples, one for each overload of ``name``."""
    signature = re.compile(rf'{re.escape(name)}\(.*\)(\s*->.*)?$')
    overloads = []
    for line in doc.splitlines():
        if signature.match(line):
            overloads.append((line, []))
        elif overloads:
            overloads[-1][1].append(line)
    return overloads


class AutoOverloads(SphinxDirective):
    required_arguments = 1

    def run(self):
        path = self.arguments[0]
        parent_path, name = path.rsplit('.', 1)
        parent = pydoc.locate(parent_path)
        obj = pydoc.locate(path)
        overloads = split_overloads(name, obj.__doc__)
        if not overloads:
            raise self.error(f'No overload signatures found in the docstring of {path}')

        if inspect.isclass(parent):
            header = ['.. py:method:: {}']
        else:
            header = ['.. py:function:: {}', f'   :module: {parent_path}']

        lines = StringList()
        for index, (signature, body) in enumerate(overloads):
            for line in header:
                lines.append(line.format(signature), path)
            if index > 0:
                # All overloads share a name, so only the first can be a cross-reference target
                lines.append('   :no-index:', path)
            lines.append('', path)
            docstring = NumpyDocstring(body, self.config, self.env.app, obj=obj)
            for line in docstring.lines():
                lines.append(f'   {line}', path)
            lines.append('', path)

        container = nodes.Element()
        self.state.nested_parse(lines, self.content_offset, container)
        return container.children


def setup(app):
    app.add_directive('autooverloads', AutoOverloads)
    return {'parallel_read_safe': True}
