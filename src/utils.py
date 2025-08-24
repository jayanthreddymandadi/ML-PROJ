import os
import joblib
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    """
    Saves a Python object to the specified file path using joblib.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)


def load_object(file_path):
    """
    Loads a Python object from the specified file path using joblib.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return joblib.load(file_path)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    """
    Trains and evaluates multiple models using GridSearchCV.
    
    Returns:
        dict: model name -> (best_model, best_score)
    """
    report = {}

    for name, model in models.items():
        params = param.get(name, {})

        gs = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        score = best_model.score(X_test, y_test)

        report[name] = (best_model, score)

    return report
