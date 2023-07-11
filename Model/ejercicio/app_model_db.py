from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import sqlite3
from sklearn.linear_model import LinearRegression

#this sets the directory
os.chdir(os.path.dirname(__file__))

#this part creates the flask app
app = Flask(__name__)
app.config['DEBUG'] = True

#this part makes a route for the endpoint
@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising"

#this is the data set
data = pd.read_csv('data/Advertising.csv')

# 1. Endpoint que devuelva la predicción de los nuevos datos enviados mediante argumentos en la llamada
@app.route('/v1/predict', methods=['GET'])
def predict():
    model = pickle.load(open('data/advertising_model','rb'))

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Missing args, the input values are needed to predict"
    else:
        prediction = model.predict([[int(tv),int(radio),int(newspaper)]])
        return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'



@app.route('/v2/predict', methods=['GET'])
def predict2():
    # Get input data from the request query parameters
    TV = request.args.get('TV')
    radio = request.args.get('radio')
    newspaper = request.args.get('newspaper')

    # Check if input values are None
    if TV is None or radio is None or newspaper is None:
        return jsonify({'error': 'Missing input values'})

    try:
        # Convert input values to int
        TV = int(TV)
        radio = int(radio)
        newspaper = int(newspaper)
    except ValueError:
        return jsonify({'error': 'Invalid input values'})

    # Create a feature vector from the input data
    features = [[TV, radio, newspaper]]

    # Load the model
    model = pickle.load(open('data/advertising_model', 'rb'))

    # Load the database and get a cursor
    conn = sqlite3.connect('Advert.db')
    cursor = conn.cursor()

    # Execute a SQL query to get the table data
    cursor.execute('SELECT * FROM ad_table')

    # Fetch all the rows from the query result
    rows = cursor.fetchall()

    # Create a DataFrame from the fetched rows
    df = pd.DataFrame(rows, columns=['TV', 'radio', 'newspaper', 'sales'])

    # Drop the 'sales' column as it is the target variable
    X = df.drop('sales', axis=1)

    

    # Make predictions using the loaded model
    predictions = model.predict(X)

    # Return the predictions as a JSON response
    return jsonify({'prediction': predictions[0]})

# @app.route('/v2/ingest_data', methods=['POST'])
@app.route('/v2/ingest_data', methods=['POST'])
def ingest_data():
    # Get the data from the request body
    data = request.get_json()

    # Verify if the required data is present
    if 'TV' not in data or 'radio' not in data or 'newspaper' not in data or 'sales' not in data:
        return jsonify({'error': 'Invalid data format'})

    try:
        # Extract the data fields
        TV = int(data['TV'])
        radio = int(data['radio'])
        newspaper = int(data['newspaper'])
        sales = int(data['sales'])
    except ValueError:
        return jsonify({'error': 'Invalid data format'})

    # Connect to the database
    conn = sqlite3.connect('Advert.db')
    cursor = conn.cursor()

    # Insert the data into the database
    cursor.execute('INSERT INTO ad_table (TV, radio, newspaper, sales) VALUES (?, ?, ?, ?)',
                   (TV, radio, newspaper, sales))

    # Commit the changes and close the database connection
    conn.commit()
    conn.close()

    # Return a success response
    return jsonify({'message': 'Data ingested successfully'})


# @app.route('/v2/retrain', methods=['PUT'])
@app.route('/v2/retrain', methods=['PUT'])
def retrain_model():
    # Load the database and get a cursor
    conn = sqlite3.connect('Advert.db')
    cursor = conn.cursor()

    # Execute a SQL query to get the table data
    cursor.execute('SELECT * FROM ad_table')

    # Fetch all the rows from the query result
    rows = cursor.fetchall()

    # Create a DataFrame from the fetched rows
    df = pd.DataFrame(rows, columns=['TV', 'radio', 'newspaper', 'sales'])

    # Close the database connection
    conn.close()

    # Split the data into features (X) and target variable (y)
    X = df.drop('sales', axis=1)
    y = df['sales']

    # Train a new model using the updated data
    model = LinearRegression()
    model.fit(X, y)

    # Save the trained model to a file
    pickle.dump(model, open('data/advertising_model', 'wb'))

    # Return a success response
    return jsonify({'message': 'Model retrained successfully'})

app.run()

