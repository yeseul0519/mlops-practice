import mlflow.sklearn


model_uri = "models:/iris-classifier@champion"

model = mlflow.sklearn.load_model(model_uri)

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("prediction:", prediction)