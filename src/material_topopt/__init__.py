"""The material topopt package."""
from __future__ import print_function
import os

FILE_DIRECTORY = os.path.dirname(__file__)
HOME_DIRECTORY = os.path.abspath(os.path.join(FILE_DIRECTORY, "../../"))
MESH_EXAMPLES = os.path.abspath(os.path.join(HOME_DIRECTORY, "mesh_files"))
DOCS = os.path.abspath(os.path.join(HOME_DIRECTORY, "docs"))

__all__ = ["HOME_DIRECTORY", "MESH_EXAMPLES", "DOCS"]
