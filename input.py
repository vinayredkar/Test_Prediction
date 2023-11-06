from flask import Flask, request, render_template
import numpy as np
import xgboost as xgb

def create_app():
    app = Flask(__name__)

    # Load your trained XGBoost model here
    # Replace 'your_model.pkl' with the actual path to your trained model file.
    xgbr = xgb.XGBRegressor()
    xgbr.load_model('D:/Sem 7/Major project/xgboost_model.pkl')

    def calculate_sales(data):
        # Calculate sales using your model
        return max(0, xgbr.predict(data)[0])

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Get input values from the form
            Power = float(request.form['Power'])
            ratings = float(request.form['ratings'])
            ton = float(request.form['ton'])
            stars = float(request.form['stars'])
            copper = int(request.form['copper'])
            inverter = int(request.form['inverter'])
            split = int(request.form['split'])
            discount_price = float(request.form['discount_price'])
            actual_price = float(request.form['actual_price'])

            # Prepare the input data for prediction
            input_data = np.array([[Power, ratings, ton, stars, copper, inverter, split, discount_price, actual_price]])

            # Calculate sales using the model
            sales = calculate_sales(input_data)

            return render_template('index.html', sales=sales)
        except:
            return render_template('index.html', error="Please enter valid input values.")

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
