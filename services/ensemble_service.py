import joblib
import numpy as np

class EnsembleService:

    def __init__(self):
        self.model = joblib.load("Random Forest_ensemble_methods.pkl")

    def predict(self, *args):

        if len(args) != 14:
            raise ValueError("The number of features is incorrect. Expected 14 features.")
        
        # convert the args tuple to an np array
        features = np.array([args])

        # prepare data
        # features = np.array([['latitude', 'longitude', 'air_quality_index', 'temperature', 'humidity', 'traffic_density', 'average_speed', 'population_density', 'building_density', 'congestion_level', 'vehicle_type_encoded', 'hour', 'day', 'month']])

        accident_risk = self.model.predict(features)[0]