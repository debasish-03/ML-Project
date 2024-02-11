import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    print("File Path: ", str(file_path))
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, Y_train, X_test, Y_test, models: dict):
    print("Evaluate model started")
    try:
        report = {}

        for model_name, model in models.items():
            ml_model = model.fit(X_train, Y_train)

            Y_train_pred = ml_model.predict(X_train)
            Y_test_pred = ml_model.predict(X_test)

            train_model_score = r2_score(Y_train, Y_train_pred)
            test_model_score = r2_score(Y_test, Y_test_pred)

            report[model_name] = test_model_score

        return report
            
    except Exception as e:
        raise CustomException(e, sys)