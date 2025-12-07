import sys
from pathlib import Path
from pkgutil import extend_path

# Ensure './src' is on sys.path so that 'src/keyword_lab' is discoverable
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Turn this into a namespace package that can span multiple locations
__path__ = extend_path(__path__, __name__)

from .cli import app
