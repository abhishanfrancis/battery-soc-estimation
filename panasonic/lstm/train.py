from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parent))

from training import main

if __name__ == "__main__":
    main("LSTM", DIRECTORY)
