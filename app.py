import pandas as pd
import pickle
from flask import Flask, request
from sklearn.preprocessing import LabelEncoder as le
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


@app.route('/get_diamond_price')
def get_diamond_price():
    carat = request.args.get("carat")
    cut = request.args.get("cut")
    color = request.args.get("color")
    clarity = request.args.get("clarity")
    x = request.args.get("x")
    y = request.args.get("y")
    z = request.args.get("z")
    df = pd.DataFrame([{'carat': carat, 'cut':cut, 'color':color, 'clarity':clarity, 'x':x, 'y':y, 'z':z}])
    model = pickle.load(open('static/saved_models/randomforestregressor.pkl', 'rb'))
    cut_le = pickle.load(open('static/saved_models/cut_le.pkl', 'rb'))
    color_le= pickle.load(open('static/saved_models/color_le.pkl', 'rb'))
    clarity_le = pickle.load(open('static/saved_models/clarity_le.pkl', 'rb'))
    df['cut'] = cut_le.transform(df['cut'])
    df['color'] = color_le.transform(df['color'])
    df['clarity'] = clarity_le.transform(df['clarity'])
    result = model.predict(df)[0]
    return  {"result": result, "input":{'carat': carat, 'cut':cut, 'color':color, 'clarity':clarity, 'x':x, 'y':y, 'z':z}}


@app.route('/train_model')
def train_model():
    df = pd.read_csv('static/datasets/diamonds.csv')
    cut_le, color_le, clarity_le = le(), le(), le()
    df['cut'] = cut_le.fit_transform(df['cut'])
    df['color'] = color_le.fit_transform(df['color'])
    df['clarity'] = clarity_le.fit_transform(df['clarity'])
    pickle.dump(cut_le, open('static/saved_models/cut_le.pkl', 'wb'))
    pickle.dump(color_le, open('static/saved_models/color_le.pkl', 'wb'))
    pickle.dump(clarity_le, open('static/saved_models/clarity_le.pkl', 'wb'))
    df.drop(['depth', 'table'], axis=1, inplace=True)
    X = df.drop('price', axis=1) 
    y = df.price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model= RandomForestRegressor()
    model = model.fit(X_train, y_train)
    pickle.dump(model, open('static/saved_models/randomforestregressor.pkl', 'wb'))
    return 'model was built sucessfully'




