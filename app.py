
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open("model/ridge_classifier.pkl", "rb") as file:
    model = pickle.load(file)

# Load TF-IDF vectorizer
with open("model/tfidf_vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        ticket = request.form["ticket"]

        # Transform ticket using trained TF-IDF
        ticket_tfidf = tfidf.transform([ticket])

        # Predict category
        prediction = model.predict(ticket_tfidf)[0]

        return render_template(
            "result.html",
            ticket=ticket,
            prediction=prediction
        )

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)

