import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print('Before Loading')
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            print('After loading')
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 sex : str,
                 age : int,
                 smoker: str,
                 bmi: int,
                 children: int,
                 region: str):
        self.sex = sex

        self.age = age

        self.smoker = smoker

        self.bmi = bmi

        self.children = children

        self.region = region

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'sex': [self.sex],
                'age': [self.age],
                'smoker': [self.smoker],
                'bmi': [self.bmi],
                'children': [self.children],
                'region': [self.region]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)