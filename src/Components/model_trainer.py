import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "CatBoostClassifier": CatBoostClassifier(verbose=False),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [5, 10, 20],
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200],
                },
                "Logistic Regression": {},
                "XGBClassifier": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200],
                },
                "CatBoostClassifier": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [50, 100],
                },
                "AdaBoostClassifier": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200],
                },
                "KNeighborsClassifier": {"n_neighbors": [3, 5, 7]},
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            # Select best model
            best_model_name, (best_model, best_model_score) = max(
                model_report.items(), key=lambda x: x[1][1]
            )

            if best_model_score < 0.6:
                raise CustomException(
                    "No suitable model found with acceptable accuracy."
                )

            logging.info(
                f"Best model found: {best_model_name} with score {best_model_score:.2f}"
            )

            # Save best trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Evaluate on test
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            return f"Model: {best_model_name}\nAccuracy: {accuracy:.2f}\n\n{report}"

        except Exception as e:
            raise CustomException(e, sys)
