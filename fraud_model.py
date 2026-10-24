"""Train a fraud classifier and predict single transactions with input validation."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from data_quality import clean

FEATURES = ["amount", "hour", "txn_last_24h", "new_receiver", "device_changed", "location_mismatch"]

def train(path="upi_transactions.csv"):
    df = clean(pd.read_csv(path))
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["is_fraud"], test_size=0.2, random_state=42, stratify=df["is_fraud"])
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print("Precision:", round(precision_score(y_test, pred), 3))
    print("Recall   :", round(recall_score(y_test, pred), 3))
    return model

def validate_transaction(txn: dict):
    """Return (True, '') if valid, else (False, error message)."""
    for f in FEATURES:
        if f not in txn or txn[f] is None or txn[f] == "":
            return False, f"Missing field: {f}"
        if not isinstance(txn[f], (int, float)) or isinstance(txn[f], bool):
            return False, f"Invalid type for {f}"
    if txn["amount"] <= 0:
        return False, "Amount must be greater than 0"
    if not 0 <= txn["hour"] <= 23:
        return False, "Hour must be between 0 and 23"
    if txn["txn_last_24h"] < 0:
        return False, "Transaction count cannot be negative"
    for f in ("new_receiver", "device_changed", "location_mismatch"):
        if txn[f] not in (0, 1):
            return False, f"{f} must be 0 or 1"
    return True, ""

def predict(model, txn: dict):
    ok, msg = validate_transaction(txn)
    if not ok:
        return {"status": "error", "message": msg}
    label = int(model.predict(pd.DataFrame([txn])[FEATURES])[0])
    return {"status": "ok", "prediction": "Fraud" if label else "Genuine"}

if __name__ == "__main__":
    train()
