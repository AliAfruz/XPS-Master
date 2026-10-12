"""Development runner for the XPS-master desktop application.

The project remains a Python application during development. A Windows .exe is
deliberately deferred until the desktop application and its validation suite are
complete.
"""

from pathlib import Path
import sys

# Always launch the checkout beside this runner, including when another
# development checkout has been installed in the selected Python environment.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from xps_afruz.ui.app import main


if __name__ == "__main__":
    raise SystemExit(main())
