from pathlib import Path
PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
RANDOM_STATE = 42