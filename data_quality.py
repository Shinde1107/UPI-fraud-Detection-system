"""Data-quality checks: missing values, duplicates, invalid values."""
import pandas as pd

def check(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "negative_or_zero_amount": int((df["amount"] <= 0).sum()),
        "invalid_hour": int((~df["hour"].between(0, 23)).sum()),
    }

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna()
    df = df[(df["amount"] > 0) & df["hour"].between(0, 23)]
    return df.reset_index(drop=True)

if __name__ == "__main__":
    raw = pd.read_csv("upi_transactions.csv")
    print("Before cleaning:", check(raw))
    print("After cleaning :", check(clean(raw)))
