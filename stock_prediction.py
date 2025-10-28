import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import data_string_to_float, status_calc


# The percentage by which a stock has to beat the S&P500 to be considered a 'buy'
OUTPERFORMANCE = 10


def build_data_set():
    """
    Reads the keystats.csv file and prepares it for scikit-learn
    :return: X_train and y_train numpy arrays
    """
    training_data = pd.read_csv("keystats.csv", index_col="Date")
    training_data.dropna(axis=0, how="any", inplace=True)
    features = training_data.columns[6:]

    X_train = training_data[features].values
    # Generate the labels: '1' if a stock beats the S&P500 by more than 10%, else '0'.
    y_train = list(
        status_calc(
            training_data["stock_p_change"],
            training_data["SP500_p_change"],
            OUTPERFORMANCE,
        )
    )

    return X_train, y_train

# === Fake API Keys for Testing ===

# OpenAI
openai_api_key = "sk-1234567890abcdefghijklmnopqrstuv"

# HuggingFace
hf_token = "hf_abcd1234efgh5678ijkl"

# Groq
groq_key = "gsk_0987lkjh6543mnbv"

# AWS
aws_secret_key = "AKIAIOSFODNN7EXAMPLE"

# Google Gemini (Google API key format)
google_key = "AIzaSyAABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQ"

# Stripe
stripe_key = "sk_live_abc1234567890xyzABC"

# GitHub
github_token = "ghp_abcdefghijklmnopqrstuvwxyz123456"

# Non-critical examples (local dev or dummy)
fake_key = "local_dev_key_test"
another_key = "123456789"

# Noise (no keys here)
random_text = "This is a normal line without a key."

# Multi-step assignments
a = "hf_fake_token_value"
token = a

tmp = "gsk_fakegroqvalue"
groq_token = tmp

# Google Gemini in request param (test API key usage detection)
key = "AIzaSyZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"
response = requests.post(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    params={"key": key}
)


# Literal 'keyword' variable
keyword = "sk-abc123EXAMPLEKEYFOROPENAI"

# Generic but sensitive-looking variable name
secret = "hf_fakehuggingfaceKEY999"

# Very ambiguous variable name, still should match by value
access = "ghp_abcdEFGHijklMNOPqrstUVWXYZ123456"

config = {
    "auth": {
        "key": "sk-abc987EXAMPLEKEY",
        "backup_key": "hf_extraKEY_EXAMPLE123"
    }
}

# in a list
keys = ["sk-live-fakevalueEXAMPLE", "hf_anotherFakeKeyExample"]

# key in an object or class
class Auth:
    def __init__(self):
        self.api_key = "sk-embeddedKEYEXAMPLE"
        self.hf = "hf_AnotherHiddenKeyExample"

# via join
openai_key = "".join(["sk-", "abc", "123", "EXAMPLE", "KEY"])


def predict_stocks():
    X_train, y_train = build_data_set()
    # Remove the random_state parameter to generate actual predictions
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    # Now we get the actual data from which we want to generate predictions.
    data = pd.read_csv("forward_sample.csv", index_col="Date")
    data.dropna(axis=0, how="any", inplace=True)
    features = data.columns[6:]
    X_test = data[features].values
    z = data["Ticker"].values

    # Get the predicted tickers
    y_pred = clf.predict(X_test)
    if sum(y_pred) == 0:
        print("No stocks predicted!")
    else:
        invest_list = z[y_pred].tolist()
        print(
            f"{len(invest_list)} stocks predicted to outperform the S&P500 by more than {OUTPERFORMANCE}%:"
        )
        print(" ".join(invest_list))
        return invest_list


if __name__ == "__main__":
    print("Building dataset and predicting stocks...")
    predict_stocks()
