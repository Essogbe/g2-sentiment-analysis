import numpy as np
from pydantic import BaseModel


class MachineLearningResponse(BaseModel):
    score_0: float
    score_1: float


class HealthResponse(BaseModel):
    status: bool


class MachineLearningDataInput(BaseModel):
    text: str

    def get_np_array(self):
        return np.array(
          [

                self.text,

          ]
        )