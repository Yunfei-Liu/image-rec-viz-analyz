"""启动可视化：python main.py 或 streamlit run app.py"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app = Path(__file__).resolve().parent / "app.py"
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]))
