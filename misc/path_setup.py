import os
import sys

path_current = os.path.dirname(__file__)
path_parent = os.path.abspath(os.path.join(path_current, ".."))
sys.path.append(path_parent)
