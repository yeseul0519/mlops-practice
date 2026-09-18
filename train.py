import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


mlflow.set_experiment("mlops-practice")

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

n_estimators = 50
max_depth = 3

run_name = f"rf_depth{max_depth}_trees{n_estimators}"

with mlflow.start_run(run_name=run_name):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print(f"accuracy: {accuracy:.4f}")