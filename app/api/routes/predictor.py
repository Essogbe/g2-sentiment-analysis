import json
from fastapi import APIRouter, HTTPException

from core.config import INPUT_EXAMPLE
from services.predict import ModelHandlerScore
from app.api.schemas.prediction import (
    MachineLearningResponse,
)

router = APIRouter()

model = ModelHandlerScore(
    model_path="app/models",
    tokenizer_path="app/models",
)


## Change this portion for other types of models
## Add the correct type hinting when completed
def get_prediction(text_to_analyse):
    sentiment, _ = model.predict(text_to_analyse)
    
    return sentiment


@router.post(
    "/predict",
    response_model=MachineLearningResponse,
    name="predict:get-data",
)
async def predict(data_input: str):

    if not data_input:
        raise HTTPException(status_code=404, detail="Aucun texte entré!")
    try:
        prediction = get_prediction(data_input)

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return MachineLearningResponse(prediction=prediction)


# @router.get(
#     "/health",
#     response_model=HealthResponse,
#     name="health:get-data",
# )
# async def health():
#     is_health = False
#     try:
#         test_input = MachineLearningDataInput(
#             **json.loads(open(INPUT_EXAMPLE, "r").read())
#         )
#         test_point = test_input.get_np_array()
#         get_prediction(test_point)
#         is_health = True
#         return HealthResponse(status=is_health)
#     except Exception:
#         raise HTTPException(status_code=404, detail="Unhealthy")
