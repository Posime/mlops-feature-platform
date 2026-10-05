import time

from feast import FeatureStore


def test_online_read(user_id=1015):
    store = FeatureStore(repo_path="src/features")

    start_time = time.perf_counter()
    response = store.get_online_features(
        features=[
            "user_credit_features:transaction_amount",
            "user_credit_features:account_balance",
            "user_credit_features:credit_score",
            "user_credit_features:failed_transactions_24h",
        ],
        entity_rows=[{"user_id": user_id}],
    ).to_dict()
    latency_ms = (time.perf_counter() - start_time) * 1000

    print(f"⚡ [ONLINE FEATURE STORE] Fetched record for user_id={user_id} in {latency_ms:.2f} ms")
    print(response)


if __name__ == "__main__":
    test_online_read()
