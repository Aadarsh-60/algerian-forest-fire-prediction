# Algerian Forest Fire Regression Model

This project is a machine learning web application that predicts the Fire Weather Index (FWI) for Algerian forest fire data. It uses a trained Ridge Regression model and a Flask frontend where users can enter weather and fire index values to get a prediction.

## Project Overview

Forest fires are strongly affected by weather conditions such as temperature, humidity, wind speed, and rainfall. This project uses the Algerian Forest Fires dataset to build a regression model that estimates the fire risk index from environmental features.

The application includes:

- Data cleaning and exploratory data analysis notebooks
- Feature engineering and model training notebook
- Trained Ridge Regression model saved as a pickle file
- Standard scaler saved as a pickle file
- Flask web application for real-time prediction

## Tech Stack

- Python
- Flask
- NumPy
- Pandas
- Scikit-learn
- HTML and CSS

## Project Structure

```text
ALGERIAN_FOREST_FIRE_REGRESSION_MODEL/
├── app.py
├── requirement.txt
├── README.md
├── MODELS/
│   ├── ridge.pkl
│   └── scaler.pkl
├── NOTEBOOK/
│   ├── Algerian_forest_fires_dataset_UPDATE.csv
│   ├── Algerian_forest_fires_cleaned_dataset.csv
│   ├── 2.0-EDA And FE Algerian Forest Fires.ipynb
│   └── 3.0-Model Training.ipynb
└── templates/
    ├── home.html
    └── index.html
```

## Input Features

The model takes the following input values:

- Temperature
- RH: Relative Humidity
- Ws: Wind Speed
- Rain
- FFMC: Fine Fuel Moisture Code
- DMC: Duff Moisture Code
- ISI: Initial Spread Index
- Classes: Fire or not fire class value
- Region: Region value

## Output

The application predicts the Fire Weather Index (FWI), which indicates forest fire risk based on the entered conditions.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/algerian-forest-fire-regression.git
cd algerian-forest-fire-regression
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirement.txt
```

## Run the Application

Start the Flask app:

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

## Model Information

The trained model is stored in:

```text
MODELS/ridge.pkl
```

The standard scaler used during training is stored in:

```text
MODELS/scaler.pkl
```

Both files are loaded automatically when the Flask application starts.

## Workflow

1. Load and clean the Algerian Forest Fires dataset.
2. Perform exploratory data analysis and feature engineering.
3. Train regression models.
4. Save the best Ridge Regression model and scaler.
5. Build a Flask app to collect user input and return predictions.

## Future Improvements

- Add model performance metrics to the web interface
- Improve input validation and error messages
- Add more model comparison details
- Deploy the app on Render, Railway, or another cloud platform

## Author

Created by Aadarsh.
