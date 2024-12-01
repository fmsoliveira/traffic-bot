import joblib
import numpy as np

class ClassificationService:

    def __init__(self):
        self.model = joblib.load('AI_training_files/Logistic Regression_classification.pkl')

    def predict(self, congestion_level, traffic_density, population_density, building_density):
        # prepare data
        features = np.array([[congestion_level, traffic_density, population_density, building_density]])

        # predict
        accident_risk = self.model.predict(features)[0]

        return accident_risk