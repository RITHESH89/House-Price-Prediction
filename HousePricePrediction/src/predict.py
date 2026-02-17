
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

def predict_price(overall_qual, gr_liv_area, garage_cars, total_bsmt_sf):
    features = np.array([[overall_qual, gr_liv_area, garage_cars, total_bsmt_sf]])
    prediction = model.predict(features)
    return prediction[0]

if __name__ == "__main__":
    price = predict_price(5, 1500, 1, 800)
    print("Predicted Price:", price)
