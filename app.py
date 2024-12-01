from flask import Flask, jsonify, render_template

from controllers.home_controller import home_bp  # Import the home controller
from controllers.api_controller import api_bp

app = Flask(__name__)


# Register the blueprint for the home route
app.register_blueprint(home_bp)

app.register_blueprint(api_bp)


if __name__ == '__main__':
    app.run(debug=True)
