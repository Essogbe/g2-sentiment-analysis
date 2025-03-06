import json
from typing import Any

from fastapi import APIRouter, HTTPException
import requests
from app.core.config import INPUT_EXAMPLE,HF_TOKEN,HF_API_URL
from app.services.predict import MachineLearningModelHandlerScore
from loguru import logger

from app.api.schemas.prediction import (
    MachineLearningResponse,
MachineLearningDataInput,
HealthResponse
)

router = APIRouter()
"""
model = MachineLearningModelHandlerScore(
    model_path="app/models",
    tokenizer_path="app/models",
)
"""
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
    logger.info("Post Request on HF Initialized")
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    logger.info(f"Post Request finished with response status {response.status_code}")
    return response.json(),response.status_code



## Change this portion for other types of models
## Add the correct type hinting when completed
def get_prediction(text_to_analyse) -> tuple[float,float]:
    output,status = query({
        "inputs": text_to_analyse,
    })
    
    if status==200:
        pred=output[0]
        pred.sort(key=lambda x: x['label'], reverse=False)
        return pred[0]['score'],pred[1]['score']
    else:
        raise Exception(f"Error occured during prediction {output['error']}")


@router.post(
    "/predict",
    response_model=MachineLearningResponse,
    name="predict:get-data",
)
async def predict(data_input: MachineLearningDataInput):
    logger.info(f"Starting prediction on <{data_input.text}>")

    if not data_input.text:
        raise HTTPException(status_code=404, detail="Aucun texte entré!")
    try:
        prediction = get_prediction(data_input.text)

    except Exception as err:
        logger.error(f"Error occured during prediction {err} ")
        raise HTTPException(status_code=500, detail=f"Exception: {err}")


    logger.info(f"Prediction success {prediction}")
    return MachineLearningResponse(**{"score_0":prediction[0],"score_1":prediction[1]})


@router.get(
    "/health",
    response_model=HealthResponse,
    name="health:get-data",
)
async def health():
    is_health = False
    logger.info("Starting Healthcheck ...")
    try:
        test_input = MachineLearningDataInput(
            **json.loads(open(INPUT_EXAMPLE, "r").read())
        )
        test_point = test_input.get_np_array()

        get_prediction(test_point[0])
        logger.info("Healthcheck success : application on")
        is_health = True
        return HealthResponse(status=is_health)
    except Exception as err:
        logger.error(f"Error during healthcheck {err}")
        raise HTTPException(status_code=404, detail=err)
