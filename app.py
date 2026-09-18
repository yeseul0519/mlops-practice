import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
import time

from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response


MODEL_URI = "./deployment_model"

app = FastAPI(
    title="Iris Classifier API",
    version="1.0"
)

PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds"
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
    start_time = time.time()

    PREDICTION_COUNT.inc()

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    latency = time.time() - start_time
    PREDICTION_LATENCY.observe(latency)

    return {"prediction": int(prediction[0])}

@app.get("/health")
def health():
    return {"status": "healthy"}



@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )