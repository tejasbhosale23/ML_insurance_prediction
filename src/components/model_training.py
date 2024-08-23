import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models =  {
                'Random Forest Regressor': RandomForestRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'XGBRegressor': XGBRegressor(),
                'CatBoosting Regressor': CatBoostRegressor(verbose=False),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'Support Vector Regressor': SVR()
            }

            params = {

                'Random Forest Regressor': {
                    'max_depth': [1,50,100,150],
                    'n_estimators': [8,16,32,64,128]
                },
                'Decision Tree' : {
                    'criterion': ['squared_error', 'poisson', 'absolute_error'],
                    'max_depth': [1,2,3,4,6,8],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2']
                },
                'Gradient Boosting': {
                    'n_estimators': [50,100,150,200],
                    'learning_rate': [0.001, 0.1, 1,1.5, 2,2.5],
                    'ccp_alpha': [1,2]
                },
                'Linear Regression': {},
                'XGBRegressor': {
                    'n_estimators': [100,200,300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                },
                'CatBoosting Regressor': {
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations':[30,50,100]
                },
                'AdaBoostRegressor': {
                    'learning_rate': [.1, .01, .5, .001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Support Vector Regressor': {
                    'C': [1,2,3,4,10,50,100],
                    'gamma': [0.1, 0.2, 0.001],
                    'epsilon': [0.1, 0.001, 0.2, 0.3]
                }
            }

            model_report: dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)    

            # to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info(f'best found model on both training and testing dataset.')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)