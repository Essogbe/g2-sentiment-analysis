import pytest
from app.models.prediction import MachineLearningDataInput, MachineLearningResponse, HealthResponse

def test_machine_learning_data_input():
    data_input = MachineLearningDataInput(text="sample text")
    np_array = data_input.get_np_array()
    assert np_array[0] == "sample text"

def test_machine_learning_response():
    response = MachineLearningResponse(score_0=0.5, score_1=0.7)
    assert response.score_0 == 0.5
    assert response.score_1 == 0.7

def test_health_response():
    response = HealthResponse(status=True)
    assert response.status is True