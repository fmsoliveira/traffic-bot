from flask import Blueprint, render_template
from services.map_service import MapService

# Create a Blueprint for the home route
home_bp = Blueprint('home', __name__, static_folder='static')

# Home route
@home_bp.route('/')
def home():
    # Call a service function to generate the map
    map_service_instance = MapService()
    map_html = map_service_instance.create_map()
    return render_template('index.html', map_html=map_html)
    #return render_template('index.html')

@home_bp.route('/accident-predict')
def accidents_predict():    
    return render_template('predict.html')

@home_bp.route('/accident-predict-random-forest')
def accident_predict_random_forest():
    return render_template('predict-random-forest.html')