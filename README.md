# Bio-Implant Expert System for Magnesium Alloys

## Overview
The **Multi-Parameter Expert Recommendation System** is an automated Machine Learning (ML) pipeline and web application designed for the predictive assessment of biodegradable magnesium implants. Traditionally, evaluating orthopedic magnesium alloys relies on manual, fragmented analysis of friction, corrosion, and wear data. This system integrates distinct datasets—Coefficient of Friction (COF), Open Circuit Potential (OCP), and 3D Profilometry (Wear Depth)—into a unified environment to provide real-time clinical decision support.

## Features
* **Automated Data Processing:** Standardizes inputs from various experimental datasets (Pure Mg, Al-Mg-Bi, Al-Mg-Sr, Al-Mg-Zn).
* **Predictive ML Models:** Utilizes Random Forest regression to forecast the nonlinear, time-dependent tribocorrosion behavior of Mg alloys.
* **Model Evaluation & Comparison:** Includes in-depth comparative analysis of different machine learning models to ensure high-accuracy friction predictions.
* **Expert Recommendation Engine:** Translates predicted numerical values into accessible, color-coded insights (e.g., assessing if friction is too high or if the corrosion rate is stable).
* **Interactive Web Interface:** A Flask-based frontend allowing users to select an alloy, input an exposure time, and receive instant performance metrics.

## File Structure
* `expert_system_backend.py`: The training engine. Processes the raw .csv data and trains the Random Forest regressors, saving them as .pkl files.
* `app.py`: The Flask inference engine. Loads the trained models and serves the backend logic for the web interface.
* `index.html`: The frontend user interface (HTML/CSS/JS).
* `Machine_Learning_Model_Comparison_for_COF_Prediction_PDF.ipynb`: A Jupyter Notebook containing exploratory data analysis, ML model performance comparisons for COF prediction, and code to generate high-quality Actual vs. Predicted PDF plots.
* `alloys_database_generated.csv`, `Friction_File.csv`, `OCP.csv`, `Wear profile...csv`: Raw experimental data used to train the digital twin models.

## How to Run the Project Locally

### 1. Prerequisites
Ensure you have Python installed on your computer. Install the required libraries by opening your terminal or command prompt and running:

`pip install flask pandas scikit-learn joblib numpy matplotlib seaborn`

*(Note: `matplotlib` and `seaborn` are required if you plan to run the Jupyter Notebook).*

### 2. Train the Models
Before running the web app, you need to generate the machine learning models. Run the backend script:

`python expert_system_backend.py`

*(This will read the CSV files and generate `random_forest_model.pkl`, `ocp_model.pkl`, and `wear_database.pkl` in your folder).*

### 3. Start the Web Application
Run the Flask application:

`python app.py`

### 4. View the App
Open your web browser and go to `http://127.0.0.1:5000/` to interact with the Expert System.

### 5. (Optional) View Model Analysis
To explore the detailed performance comparison of different ML models for friction prediction, open the `Machine_Learning_Model_Comparison_for_COF_Prediction_PDF.ipynb` file using Jupyter Notebook, JupyterLab, or Google Colab.

## Research Context
This tool was developed to overcome the "data fragmentation" barrier in the clinical use of degradable magnesium alloys. By establishing a digital twin of the alloy's degradation profile, it successfully identifies high-performers (e.g., Al-Mg-Sr as a "Tribological Leader" and Al-Mg-Zn as an "Electrochemical Leader").
