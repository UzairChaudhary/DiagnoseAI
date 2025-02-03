# Disease Prediction API

This project is a FastAPI application for predicting diseases based on an array of symptoms provided as input. The application utilizes machine learning models (Random Forest, Naive Bayes, and SVM) to make predictions.

## Project Structure

```
disease-prediction-api
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   └── disease_model.py
│   ├── routers
│   │   ├── __init__.py
│   │   └── predict.py
│   └── utils
│       ├── __init__.py
│       └── model_utils.py
├── data
│   └── Training.csv
├── models
│   ├── rf_model.pkl
│   ├── nb_model.pkl
│   └── svm_model.pkl
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd disease-prediction-api
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the FastAPI application:
   ```
   uvicorn app.main:app --reload
   ```

4. Access the API documentation at `http://127.0.0.1:8000/docs`.

## Usage

To predict a disease, send a POST request to the `/predict` endpoint with a JSON body containing the symptoms. For example:

```json
{
  "symptoms": ["Itching", "Skin Rash", "Nodal Skin Eruptions"]
}
```

The response will include the predicted disease as a string.

## Models

The application uses the following machine learning models:
- Random Forest
- Naive Bayes
- Support Vector Machine (SVM)

These models are trained on the data provided in `data/Training.csv` and are serialized in the `models` directory.

## License

This project is licensed under the MIT License.