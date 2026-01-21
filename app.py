import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained pipeline (preprocessing + model)
pipeline = joblib.load("model/house_price_model.pkl")

# Neighborhood options (must match training dataset)
NEIGHBORHOODS = [
    "CollgCr","Veenker","Crawfor","NoRidge","Mitchel","Somerst","NWAmes","OldTown","BrkSide",
    "Sawyer","NridgHt","NAmes","SawyerW","IDOTRR","MeadowV","Edwards","Timber","Gilbert","StoneBr",
    "ClearCr","NPkVill","Blmngtn","BrDale","SWISU","Blueste"
]

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None
    error_msg = None

    if request.method == "POST":
        try:
            # Collect inputs from the form
            user_input = {
                "OverallQual": [float(request.form["OverallQual"])],
                "GrLivArea": [float(request.form["GrLivArea"])],
                "TotalBsmtSF": [float(request.form["TotalBsmtSF"])],
                "GarageCars": [float(request.form["GarageCars"])],
                "YearBuilt": [float(request.form["YearBuilt"])],
                "Neighborhood": [request.form["Neighborhood"]]
            }

            # Convert to DataFrame
            df_input = pd.DataFrame(user_input)
            print("DEBUG: Input DataFrame:\n", df_input)

            # Predict using pipeline
            predicted_price_value = pipeline.predict(df_input)[0]
            predicted_price = f"${predicted_price_value:,.2f}"

        except Exception as e:
            error_msg = str(e)

    return render_template(
        "index.html",
        predicted_price=predicted_price,
        error_msg=error_msg,
        neighborhoods=NEIGHBORHOODS
    )

if __name__ == "__main__":
    app.run(debug=True)
