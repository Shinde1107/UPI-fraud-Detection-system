"""Generate a synthetic UPI transaction dataset (no real user data)."""
import numpy as np
import pandas as pd

def generate(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "amount": np.round(rng.lognormal(6.0, 1.0, n), 2),
        "hour": rng.integers(0, 24, n),
        "txn_last_24h": rng.poisson(2, n),
        "new_receiver": rng.integers(0, 2, n),
        "device_changed": rng.integers(0, 2, n),
        "location_mismatch": rng.integers(0, 2, n),
    })
    # Fraud is more likely with high amount, night hours, new receiver, device/location change
    score = (
        0.0002 * df["amount"]
        + 1.6 * ((df["hour"] <= 5) | (df["hour"] >= 23))
        + 1.5 * df["new_receiver"]
        + 1.8 * df["device_changed"]
        + 1.7 * df["location_mismatch"]
        + 0.3 * df["txn_last_24h"]
        - 6.4
    )
    df["is_fraud"] = ((score + rng.normal(0, 0.6, n)) > 0).astype(int)
    # Add a few duplicates and missing values on purpose, to practise data-quality checks
    df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)
    df.loc[df.sample(20, random_state=2).index, "amount"] = np.nan
    return df

if __name__ == "__main__":
    data = generate()
    data.to_csv("upi_transactions.csv", index=False)
    print("Saved upi_transactions.csv:", data.shape)
