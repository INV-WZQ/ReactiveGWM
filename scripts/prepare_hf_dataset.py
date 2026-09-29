"""Download selected v2 samples and produce a training/inference CSV."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    from selective_agency.runtime.hf_data import main

    main()
