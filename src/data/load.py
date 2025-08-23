import pandas as pd
from src import config as C

def load_raw(split: str) -> pd.DataFrame:
    p = C.RAW / f"{split}.csv"   # split in {"train","test"}
    return pd.read_csv(p)