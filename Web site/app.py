# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# -------------------------------------------------
# Load model + scaler (they must be in the same folder)
# -------------------------------------------------
MODEL_PATH = 'predictor.pickle'
SCALER_PATH = 'scaler.pickle'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)


def build_feature_vector(form_data):
    """
    form_data = [gender, vip, income, children, age, attractiveness]
    Returns a 1-D numpy array ready for the model.
    """
    # 1. Convert to proper Python types
    gender = int(form_data[0])
    vip    = int(form_data[1])
    income = float(form_data[2])
    children = int(form_data[3])
    age    = int(form_data[4])
    attract = int(form_data[5])

    # 2. Build raw array (order must match training!)
    raw = np.array([[gender, vip, income, children, age, attract]])

    # 3. Scale the four numeric columns (Income, Children, Age, Attractiveness)
    to_scale = raw[:, [2, 3, 4, 5]]                 # shape (1,4)
    scaled   = scaler.transform(to_scale)          # shape (1,4)

    # 4. Assemble final feature vector
    features = [
        gender,                     # Gender
        vip,                        # PurchasedVIP
        scaled[0, 0],               # Income (scaled)
        scaled[0, 1],               # Children (scaled)
        scaled[0, 2],               # Age (scaled)
        scaled[0, 3],               # Attractiveness (scaled)
        scaled[0, 2] * scaled[0, 3], # Age_Attract
        scaled[0, 0] * vip          # Income_VIP
    ]

    return np.array(features).reshape(1, -1)   # (1, 8)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_msg  = None

    if request.method == 'POST':
        try:
            # ---- collect & validate form values ----
            gender       = request.form['gender']
            vip          = request.form['vip']
            income       = request.form['income']
            children     = request.form['children']
            age          = request.form['age']
            attractiveness = request.form['attractiveness']

            # basic sanity checks
            if not all([gender, vip, income, children, age, attractiveness]):
                raise ValueError("All fields are required.")

            # ---- build feature vector & predict ----
            feature_vec = build_feature_vector(
                [gender, vip, income, children, age, attractiveness]
            )
            pred = model.predict(feature_vec)[0]      # scalar
            prediction = round(float(pred), 2)

        except Exception as e:
            error_msg = str(e)

    # Pass both values to the template
    return render_template(
        'index.html',
        prediction=prediction,
        error_msg=error_msg
    )


if __name__ == '__main__':
    app.run(debug=True)