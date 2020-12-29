import pandas as pd

from flask import Flask, request
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

@app.route('/')
def home():
    return 'welcome to the homepage of BC ML deployment page. please know that noting is perfect here, and things needs to be improved.'

@app.route('/demo_api')
def demo_api():
    param1 = int(request.args.get("param1"))
    param2 = int(request.args.get("param2"))
    if (param1 is None or param2 is None):
        return 'params not received', 400
    return {'sum': param1+param2, 'mult': param1*param2, 'sub':abs(param1-param2)}, 200

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

@app.route('/get_features')
def get_features():
    carat = int(request.args.get("carat"))
    cut = int(request.args.get("cut"))
    color = int(request.args.get("color"))
    clarity = int(request.args.get("clarity"))
    x = int(request.args.get("x"))
    y = int(request.args.get("y"))
    z = int(request.args.get("z"))
    return  {'carat': carat, 'cut':cut, 'color':color, 'clarity':clarity, 'x':x, 'y':y, 'z':z}, 200


