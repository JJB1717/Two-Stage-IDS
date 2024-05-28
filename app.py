from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained LSTM model
lstm_model = load_model('test.h5')

# Load the encoder part of the trained autoencoder
encoder = load_model('autoencoder.h5')

# Load the training data
training_data = pd.read_csv("unsw_nb15/UNSW_NB15_training_set.csv")  # Replace with the path to your training data

# Extract features for scaling
features_to_scale = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sload', 'dload', 
                     'sinpkt', 'dinpkt', 'ct_srv_src', 'ct_dst_ltm', 'ct_src_ltm']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler with training data
scaler.fit(training_data[features_to_scale])

# Load the index.html page
@app.route('/')
def index():
    return render_template('index.html')

# Predict the result and load the result.html page
@app.route('/result', methods=['POST'])
def result():
    features = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sload', 'dload', 
                'sinpkt', 'dinpkt', 'ct_srv_src', 'ct_dst_ltm', 'ct_src_ltm']
    test_case = {}
    for feature in features:
        test_case[feature] = float(request.form[feature])
    test_df = pd.DataFrame([test_case])
    test_scaled = scaler.transform(test_df)
    
    # Encode the scaled test input using the encoder part of the trained autoencoder
    test_encoded = encoder.predict(test_scaled)
    
    # Reshape the encoded test input to match the shape expected by the LSTM model
    test_reshaped = test_encoded.reshape(test_encoded.shape[0], 1, test_encoded.shape[1])
    
    y_pred_prob = lstm_model.predict(test_reshaped)
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    if y_pred_class[0] == 1:
        prediction = 'ATTACK'
    else:
        prediction = 'NORMAL'
    return render_template('result1.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
