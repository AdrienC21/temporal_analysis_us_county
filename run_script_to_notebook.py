"""
Script based on the work of William James Handley
See: https://github.com/williamjameshandley/py2nb

Create automatically a notebook based on the python
file script_to_convert.py
script_to_convert.py is created with the file
run_create_script_for_notebook.py
"""

import argparse
import os

import nbformat.v4

ACCEPTED_CHARS = ['#-', '# -']
MARKDOWN_CHARS = ['#|', '# |']
ACCEPTED_CHARS.extend(MARKDOWN_CHARS)


def new_cell(nb, cell, markdown=False):
    """ Create a new cell
    Parameters
    ----------
    nb: nbformat.notebooknode.NotebookNode
        Notebook to write to, as produced by nbformat.v4.new_notebook()
    cell: str
        String to write to the cell
    markdown: boolean, optional, (default False)
        Whether to create a markdown cell, or a code cell
    """
    cell = cell.rstrip().lstrip()
    if cell:
        if markdown:
            cell = nbformat.v4.new_markdown_cell(cell)
        else:
            cell = nbformat.v4.new_code_cell(cell)
        nb.cells.append(cell)
    return ''


def str_starts_with(string, options):
    for opt in options:
        if string.startswith(opt):
            return True


def convert(script_name, notebook_name):
    """ Convert the python script to jupyter notebook"""
    with open(script_name) as f:
        markdown_cell = ''
        code_cell = ''
        nb = nbformat.v4.new_notebook()
        for line in f:
            if str_starts_with(line, ACCEPTED_CHARS):
                code_cell = new_cell(nb, code_cell)
                if str_starts_with(line, MARKDOWN_CHARS):
                    # find the first occurence of |
                    # and add the rest of the line to the markdown cell
                    markdown_cell += line[line.index('|') + 1:]
                else:
                    markdown_cell = new_cell(nb, markdown_cell, markdown=True)
            else:
                markdown_cell = new_cell(nb, markdown_cell, markdown=True)
                code_cell += line

        markdown_cell = new_cell(nb, markdown_cell, markdown=True)
        code_cell = new_cell(nb, code_cell)

        # notebook_name = os.path.splitext(script_name)[0] + '.ipynb'
        nbformat.write(nb, notebook_name)


def main():
    # args = parse_args()
    # convert(args.script_name)
    script_name = "script_to_convert.py"
    notebook_name = "Analysis.ipynb"
    convert(script_name, notebook_name)


if __name__ == "__main__":
    main()
