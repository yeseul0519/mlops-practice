import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_URI = "./deployment_model"

app = FastAPI(
    title="Iris Classifier API",
    version="1.0"
)


# 서버 시작 시 Champion 모델 로드
model = mlflow.sklearn.load_model(MODEL_URI)


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {
        "message": "Iris Classifier API",
        "model": MODEL_URI
    }


@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }

@app.get("/health")
def health():
    return {"status": "healthy"}