import pandas as pd
import joblib
import os

class CustomData:
    def __init__(self, Weather, Road_Condition, Time_of_Day, Traffic, Accident_Type,
                 Vehicle_Type, Accident_Reason, Latitude, Longitude):
        self.Weather = Weather
        self.Road_Condition = Road_Condition
        self.Time_of_Day = Time_of_Day
        self.Traffic = Traffic
        self.Accident_Type = Accident_Type
        self.Vehicle_Type = Vehicle_Type
        self.Accident_Reason = Accident_Reason
        self.Latitude = Latitude
        self.Longitude = Longitude

    def get_data_as_dataframe(self):
        return pd.DataFrame({
            "Weather": [self.Weather],
            "Road_Condition": [self.Road_Condition],
            "Time_of_Day": [self.Time_of_Day],
            "Traffic": [self.Traffic],
            "Accident_Type": [self.Accident_Type],
            "Vehicle_Type": [self.Vehicle_Type],
            "Accident_Reason": [self.Accident_Reason],
            "Latitude": [self.Latitude],
            "Longitude": [self.Longitude]
        })


class PredictPipeline:
    def __init__(self):
        model_path = os.path.join("artifacts", "model.pkl")
        self.model = joblib.load(model_path)

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)
