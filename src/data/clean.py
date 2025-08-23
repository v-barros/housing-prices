from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np
from src import config as C

CAT_NONE = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
            "FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",
            "PoolQC","Fence","MiscFeature","MasVnrType"]
NUM_ZERO = ["MasVnrArea","BsmtFullBath","BsmtHalfBath","BsmtFinSF1","BsmtFinSF2",
            "BsmtUnfSF","TotalBsmtSF","GarageCars","GarageArea"]
ORD_MAP = {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1}

def _ordinal_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = df[c].fillna("None").map(lambda x: ORD_MAP.get(x, 0))
    return df

def _basic_fixes(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()
    # types
    if "MSSubClass" in df: df["MSSubClass"] = df["MSSubClass"].astype(str)
    # NA semantics
    for c in CAT_NONE:
        if c in df: df[c] = df[c].fillna("None")
    for c in NUM_ZERO:
        if c in df: df[c] = df[c].fillna(0)
    # LotFrontage median by neighborhood is common; fallback to global median
    if "LotFrontage" in df:
        if "Neighborhood" in df:
            df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
                lambda s: s.fillna(s.median())
            ).fillna(df["LotFrontage"].median())
        else:
            df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
    # Ordinals
    ord_cols = [c for c in ["ExterQual","ExterCond","BsmtQual","BsmtCond",
                            "HeatingQC","KitchenQual","FireplaceQu",
                            "GarageQual","GarageCond","PoolQC"] if c in df]
    df = _ordinal_encode(df, ord_cols)
    # Simple engineered features
    if set(["TotalBsmtSF","1stFlrSF","2ndFlrSF"]).issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    if set(["YrSold","YearBuilt"]).issubset(df.columns):
        df["AgeAtSale"] = df["YrSold"] - df["YearBuilt"]
    if set(["YrSold","YearRemodAdd"]).issubset(df.columns):
        df["YearsSinceRemod"] = df["YrSold"] - df["YearRemodAdd"]
        df["IsRemodeled"] = (df["YearRemodAdd"] > df["YearBuilt"]).astype(int)
    # Target stabilization (train only)
    if is_train and "SalePrice" in df:
        df["SalePrice_log"] = (df["SalePrice"] + 1).apply(np.log)
    return df

def clean_and_write():
    import numpy as np  # local import to keep deps light
    # train
    train = pd.read_csv(C.RAW / "train.csv")
    train_clean = _basic_fixes(train, is_train=True)
    (C.PROCESSED).mkdir(parents=True, exist_ok=True)
    train_clean.to_parquet(C.PROCESSED / "train.parquet", index=False)
    # test
    test = pd.read_csv(C.RAW / "test.csv")
    test_clean = _basic_fixes(test, is_train=False)
    test_clean.to_parquet(C.PROCESSED / "test.parquet", index=False)
    print("✅ Wrote:", C.PROCESSED / "train.parquet", "and", C.PROCESSED / "test.parquet")

if __name__ == "__main__":
    clean_and_write()
