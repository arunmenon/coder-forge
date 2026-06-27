import pathlib
import sys

# Make the src/ package importable without installation (until pyproject lands in Phase 3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
