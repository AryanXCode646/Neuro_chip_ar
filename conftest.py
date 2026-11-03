import os
import sys

# Ensure the src directory is on the Python path for test imports
current_dir = os.path.dirname(__file__)
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
