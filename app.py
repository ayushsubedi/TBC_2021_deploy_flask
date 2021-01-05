import pandas as pd

from flask import Flask, request
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

@app.route('/')
def home():
    return '<html><body><title>bc</title>ml</body></html>'


@app.route('/demo2021')
def demo2021():
    return 'happy new year'


@app.route('/demo_api')
def demo_api():
    param1 = int(request.args.get("param1"))
    param2 = int(request.args.get("param2"))
    if (param1 is None or param2 is None):
        return 'params not received', 400
    return {'sum': param1+param2, 'mult': param1*param2, 'sub':abs(param1-param2)}, 200



@app.route('/get_features')
def get_features():
    carat = request.args.get("carat")
    cut = request.args.get("cut")
    color = request.args.get("color")
    clarity = request.args.get("clarity")
    x = request.args.get("x")
    y = request.args.get("y")
    z = request.args.get("z")
    return  {'carat': carat, 'cut':cut, 'color':color, 'clarity':clarity, 'x':x, 'y':y, 'z':z}, 200

















@app.route('/train_model')
def train_model():
    df = pd.read_csv('static/datasets/diamonds.csv')
    df['cut'] = preprocessing.LabelEncoder().fit_transform(df['cut'])
    df['color'] = preprocessing.LabelEncoder().fit_transform(df['color'])
    df['clarity'] = preprocessing.LabelEncoder().fit_transform(df['clarity'])
    df.drop(['depth', 'table'], axis=1, inplace=True)
    X = df.drop('price', axis=1) 
    y = df.price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model= RandomForestRegressor()
    model = model.fit(X_train, y_train)
    return 'model was built sucessfully'




