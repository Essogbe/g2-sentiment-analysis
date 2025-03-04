# import numpy as np

from pydantic import BaseModel


class MachineLearningResponse(BaseModel):
    prediction: str


# class HealthResponse(BaseModel):
#     status: bool


# class MachineLearningDataInput(BaseModel):
#     text: str

#     def get_np_array(self):
#         return np.array(
#             [
#                 [
#                     self.feature1,
#                     self.feature2,
#                     self.feature3,
#                     self.feature4,
#                     self.feature5,
#                 ]
#             ]
#         )
