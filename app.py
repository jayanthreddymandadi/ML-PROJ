from flask import Flask, render_template, request
import pandas as pd
import random  # Simulating predictions for now

# --- Placeholder classes ---
class CustomData:
    """
    Placeholder for your actual CustomData class.
    Includes all columns from your dataset.
    """
    def __init__(self, date, time, weather, road_cond, time_of_day,
                 name, accident_type, vehicle_type_1, accident_cause,
                 latitude, longitude, severity=None):
        self.date = date
        self.time = time
        self.weather = weather
        self.road_cond = road_cond
        self.time_of_day = time_of_day
        self.name = name
        self.accident_type = accident_type
        self.vehicle_type_1 = vehicle_type_1
        self.accident_cause = accident_cause
        self.latitude = latitude
        self.longitude = longitude
        self.severity = severity  # Optional — may be None for prediction

    def get_data_as_data_frame(self):
        """
        Converts the instance data into a pandas DataFrame.
        """
        custom_data_input_dict = {
            'Date': [self.date],
            'Time': [self.time],
            'Weather': [self.weather],
            'Road_Cond': [self.road_cond],
            'Time_Of_Day': [self.time_of_day],
            'Name': [self.name],
            'Accident_Type': [self.accident_type],
            'Vehicle_Type_1': [self.vehicle_type_1],
            'Accident_Cause': [self.accident_cause],
            'Latitude': [self.latitude],
            'Longitude': [self.longitude],
            'Severity': [self.severity]
        }
        return pd.DataFrame(custom_data_input_dict)


class PredictPipeline:
    """
    Placeholder for your actual prediction pipeline.
    """
    def predict(self, features):
        severities = ["Low", "Medium", "High"]
        return [random.choice(severities)]
# --- End placeholder classes ---


# Initialize Flask app
application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_accident', methods=['GET', 'POST'])
def predict_accident():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # Collect all form data
        data = CustomData(
            date=request.form.get('date'),
            time=request.form.get('time'),
            weather=request.form.get('weather'),
            road_cond=request.form.get('road_cond'),
            time_of_day=request.form.get('time_of_day'),
            name=request.form.get('name'),
            accident_type=request.form.get('accident_type'),
            vehicle_type_1=request.form.get('vehicle_type_1'),
            accident_cause=request.form.get('accident_cause'),
            latitude=request.form.get('latitude'),
            longitude=request.form.get('longitude'),
            severity=request.form.get('severity')  # Optional
        )

        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:")
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("Prediction Results:", results)

        return render_template('index.html', results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)