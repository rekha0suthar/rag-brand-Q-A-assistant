import sys
from pathlib import Path

# Make the flat project modules (chain, utils, ...) importable from tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
