import pytest
from fastapi.testclient import TestClient
from app.api.routes.predictor import router
from app.api.schemas.prediction import MachineLearningDataInput
from fastapi.exceptions import RequestValidationError,HTTPException
client = TestClient(router)

def test_predict():
    response = client.post("/predict", json={"text": "sample text"})
    assert response.status_code == 200
    assert "score_0" in response.json()
    assert "score_1" in response.json()




def test_predict_no_text():
    with pytest.raises(HTTPException) as exc_info:
        client.post("/predict", json={"text": ""})
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Aucun texte entré!"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": True}





def test_predict_invalid_data():
    with pytest.raises(RequestValidationError) as exc_info:
        client.post("/predict", json={"text": 12345})
    assert exc_info.value.errors() == [
        {
            "loc": ("body", "text"),
            "msg": "Input should be a valid string",
            "type": "string_type",
            "input": 12345
        }
    ]