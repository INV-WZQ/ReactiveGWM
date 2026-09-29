"""Encode missing VAE/T5 assets in a portable sample CSV."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from selective_agency.runtime.cache import main

    main(ROOT / "training/configs/hnm.yaml")
