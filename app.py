import mlflow
import mlflow.sklearn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Iris MLOps Level 2 API")

MODEL_NAME = "IrisRandomForestModel"

# Load latest registered model
def load_latest_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model_uri = f"models:/{MODEL_NAME}/latest"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Successfully loaded model: {model_uri}")
        return model

    except Exception as e:
        print(f"Model loading failed: {e}")
        return None


model = load_latest_model()


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {
        "message": "Iris MLOps API is running",
        "model": MODEL_NAME
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_unavailable"
    }


@app.post("/predict")
def predict(data: IrisInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Please register a model in MLflow."
        )

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    try:
        prediction = model.predict(features)

        return {
            "prediction": int(prediction[0]),
            "class_name": [
                "setosa",
                "versicolor",
                "virginica"
            ][int(prediction[0])]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
