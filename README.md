
# Ticket Routing System

An AI-powered Ticket Routing System that automatically classifies customer support tickets into the appropriate category using Natural Language Processing and Machine Learning.

## Project Overview

Customer support teams receive a large number of tickets every day. Manually routing these tickets to the correct department can be slow and inefficient.

This project solves the problem by automatically analyzing the ticket text and predicting its category.

## Solution

The system uses:

- TF-IDF for text feature extraction
- RidgeClassifier for ticket classification
- Flask for the web application
- HTML and CSS for the user interface
- Pickle for saving and loading the trained model

## Project Pipeline

```text
Customer Ticket
       ↓
Text Input
       ↓
TF-IDF Vectorization
       ↓
RidgeClassifier
       ↓
Predicted Category
       ↓
Ticket Routing
````

## Project Structure

```text
TicketRoutingSystem/
│
├── data/
│   └── tickets.csv
│
├── model/
│   ├── ridge_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── templates/
│   ├── index.html
│   ├── predict.html
│   └── result.html
│
├── static/
│   └── css/
│       └── style.css
│
├── Ticket Routing System.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Model

The text is converted into numerical features using TF-IDF.

The trained RidgeClassifier then predicts the ticket category.

The model was evaluated using:

* Macro F1 Score
* Classification Report
* Confusion Matrix
* ROC / AUC

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd TicketRoutingSystem
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Run:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Usage

1. Open the application.
2. Select "Route a Ticket".
3. Enter a customer support ticket.
4. Click "Predict Category".
5. The system returns the predicted ticket category.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF
* RidgeClassifier
* Flask
* HTML
* CSS
* Pickle

## Future Improvements

* Compare multiple classification models.
* Add confidence scores.
* Add an admin dashboard.
* Store and monitor routed tickets.
* Add automated department assignment.
* Deploy the application online.

## Author

Alaa Ahmed Al Imam

AI Engineer

Machine Learning • Deep Learning • Generative AI • LLMs • AI Agents

```

