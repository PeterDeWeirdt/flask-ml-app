import pandas as pd
from flask import Flask, jsonify, request
import pickle

# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    probabilities = model.predict_proba(data_df)[:, 0]

    # send back to browser
    output = {'probability': list(probabilities)}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(debug=True)