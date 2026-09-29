"""Train an HNM or SF3 main model; see --help."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from selective_agency.runtime.train import main

    main(ROOT / "training/configs/hnm.yaml")
