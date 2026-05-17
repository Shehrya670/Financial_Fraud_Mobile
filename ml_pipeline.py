from sklearn.base import BaseEstimator, TransformerMixin


MODEL_INPUT_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isMerchant",
]

NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isMerchant",
]

CATEGORICAL_FEATURES = ["type"]

TRANSACTION_TYPES = {"PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"}
DESTINATION_TYPES = {"CUSTOMER", "MERCHANT"}


class NoLeakageFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build serving-safe features from raw PaySim/API transaction fields."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy()

        if "isMerchant" not in frame.columns:
            if "destType" in frame.columns:
                frame["isMerchant"] = (
                    frame["destType"].fillna("CUSTOMER").str.upper() == "MERCHANT"
                ).astype(int)
            elif "nameDest" in frame.columns:
                frame["isMerchant"] = frame["nameDest"].fillna("").str.startswith("M").astype(int)
            else:
                frame["isMerchant"] = 0

        frame["type"] = frame["type"].fillna("TRANSFER").str.upper()
        for column in NUMERIC_FEATURES:
            frame[column] = frame[column].astype(float)

        return frame[MODEL_INPUT_COLUMNS]
