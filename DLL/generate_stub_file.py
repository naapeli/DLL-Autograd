from pybind11_stubgen import main

import sys
import os
from pathlib import Path

path = Path(r".")
os.add_dll_directory((path / "lib").absolute())

sys.argv = ["pybind11_stubgen", "_C", "-o", str(path.absolute())]

main()
