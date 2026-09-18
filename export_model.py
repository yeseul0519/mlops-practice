import mlflow.sklearn


MODEL_URI = "models:/iris-classifier@champion"
OUTPUT_PATH = "./deployment_model"


mlflow.sklearn.save_model(
    mlflow.sklearn.load_model(MODEL_URI),
    path=OUTPUT_PATH,
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

print(f"Model exported to: {OUTPUT_PATH}")