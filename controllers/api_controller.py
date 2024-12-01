from flask import Blueprint, jsonify, request

from services.classification_service import ClassificationService

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

    print("DATA ON REQUEST: ", data)

    return jsonify({
        'accident_risk': "api under development"
        'message': "api under development"
    })