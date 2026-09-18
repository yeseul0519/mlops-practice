import mlflow
from mlflow import MlflowClient


EXPERIMENT_NAME = "mlops-practice"
REGISTERED_MODEL_NAME = "iris-classifier"
ALIAS = "champion"

client = MlflowClient()


# 1. Experiment 찾기
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError(
        f"Experiment '{EXPERIMENT_NAME}'을 찾을 수 없습니다."
    )


# 2. accuracy가 높은 Run부터 조회
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.accuracy DESC"],
)

if runs.empty:
    raise ValueError("완료된 Run을 찾을 수 없습니다.")


# 3. Logged Model이 존재하는 가장 좋은 Run 찾기
best_run = None
best_model = None

for _, run in runs.iterrows():
    run_id = run["run_id"]

    logged_models = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
    )

    if not logged_models.empty:
        best_run = run
        best_model = logged_models.iloc[0]
        break


if best_run is None:
    raise ValueError(
        "저장된 모델이 연결된 Run을 찾을 수 없습니다."
    )


best_run_id = best_run["run_id"]
best_accuracy = best_run["metrics.accuracy"]
best_model_id = best_model["model_id"]

print("=== Candidate ===")
print(f"Run ID       : {best_run_id}")
print(f"Accuracy     : {best_accuracy:.4f}")
print(f"Logged Model : {best_model_id}")


# 4. 현재 Champion 확인
try:
    champion = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=ALIAS,
    )

    champion_run = client.get_run(champion.run_id)
    champion_accuracy = champion_run.data.metrics["accuracy"]

    print("\n=== Current Champion ===")
    print(f"Version  : {champion.version}")
    print(f"Run ID   : {champion.run_id}")
    print(f"Accuracy : {champion_accuracy:.4f}")

except Exception:
    champion = None
    champion_accuracy = None

    print("\nCurrent Champion does not exist.")


# 5. Candidate와 Champion 성능 비교
should_register = (
    champion is None
    or best_accuracy > champion_accuracy
)


if not should_register:
    print("\n=== Result ===")
    print("Candidate did not outperform Champion.")
    print("Registration skipped.")

else:
    print("\nCandidate outperformed Champion.")

    model_uri = f"models:/{best_model_id}"

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=ALIAS,
        version=model_version.version,
    )

    print("\n=== Result ===")
    print(f"Registered Model : {REGISTERED_MODEL_NAME}")
    print(f"New Version      : {model_version.version}")
    print(f"Alias            : {ALIAS}")
    print(f"Accuracy         : {best_accuracy:.4f}")