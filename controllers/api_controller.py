from datetime import datetime 
from flask import Blueprint, jsonify, request

from services.classification_service import ClassificationService
from services.ensemble_service import EnsembleService

# Create a Blueprint for the api route
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Define a simple API endpoint
@api_bp.route('/data', methods=['GET'])
def api():
    data = {
        "message": "This is a simple API",
        "status": "success"
    }
    return jsonify(data)

@api_bp.route('/accident-predict', methods=['POST'])
def api_predict():

    # Extract data from the request
    data = request.form
    congestion_level = float(data.get('congestion_level', 0))
    traffic_density = float(data.get('traffic_density', 0))
    population_density = int(data.get('population_density', 0))
    building_density = float(data.get('building_density', 0))

    # Compute prediction logic (replace with actual logic)
    trained_model = ClassificationService()
    accident_risk = trained_model.predict(congestion_level, traffic_density, population_density, building_density)

    # Return a JSON response
    return jsonify({
        'accident_risk': accident_risk,
        'message': 'Prediction successful!'
    })

@api_bp.route('/accident-predict-random-forest', methods=['POST'])
def api_predict_random_forest():
    
    data = request.form

    latitude = float(data.get('latitude', 0))
    longitude = float(data.get('longitude', 0))
    air_quality_index = float(data.get('air_quality_index', 0))
    temperature = float(data.get('temperature', 0))
    humidity = float(data.get('humidity', 0))
    traffic_density = float(data.get('traffic_density', 0))
    average_speed = float(data.get('average_speed', 0))
    population_density = float(data.get('population_density', 0))
    building_density = float(data.get('building_density', 0))
    congestion_level = float(data.get('congestion_level', 0))
    vehicle_type_encoded = data.get('vehicle_type_encoded', 0)
    date_and_time = (data.get('datetime', 0))

    parsed_date_time = datetime.strptime(date_and_time, "%Y-%m-%dT%H:%M")

    parsed_hour = parsed_date_time.hour
    parsed_day = parsed_date_time.day
    parsed_year = parsed_date_time.year

    trained_model = EnsembleService()
    accident_risk = trained_model.predict(latitude, longitude, air_quality_index, temperature, humidity, traffic_density, average_speed, population_density, building_density, congestion_level, vehicle_type_encoded, parsed_hour, parsed_day, parsed_year)

    return jsonify({
        'accident_risk': accident_risk,
        'message': 'Prediction successful!'
    })