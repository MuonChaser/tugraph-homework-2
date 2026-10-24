"""Download the public SNAP Stanford Web Graph edge list."""
from pathlib import Path
from urllib.request import urlretrieve
import gzip

DEST = Path(__file__).resolve().parents[1] / "data" / "web-Stanford.txt.gz"
URL = "https://snap.stanford.edu/data/web-Stanford.txt.gz"
DEST.parent.mkdir(parents=True, exist_ok=True)
if not DEST.exists():
    urlretrieve(URL, DEST)
with gzip.open(DEST, "rt", encoding="utf-8") as f:
    first_data = next(line for line in f if line and not line.startswith("#"))
print(f"Verified {DEST.name}: first data row = {first_data.strip()}")
