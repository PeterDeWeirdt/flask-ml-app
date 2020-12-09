import flask
from flask import request, jsonify
import json
import pandas as pd
from numpy.testing import assert_array_almost_equal
from app import app


def test_api_call():
    print('hello world')
    data_df = pd.DataFrame({'pclass': [3, 4],
                            'age': [2, 3],
                            'aibsp': [1, 2],
                            'fare': [50, 200]}, index=[0, 1])
    data = data_df.to_json()
    with app.test_client() as c:
        print(c)
        r = c.post('/', data=data)
        assert r.status_code == 200
        assert_array_almost_equal(json.loads(r.data)['results']['probability'], [0.1857743071262108, 0.16656764955456183])