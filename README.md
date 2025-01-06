
# Crop Production Prediction System

This **Streamlit application** predicts crop production based on agricultural data using a **RandomForestRegressor** model. The app provides advanced preprocessing, interactive visualizations, model performance metrics, and future crop yield predictions to support agricultural planning.

## Features
- **Data Upload**: Upload a FAOSTAT dataset in CSV format for analysis.
- **Data Cleaning**: Handles missing values, removes outliers, and log-transforms production data.
- **Interactive Visualizations**: Includes scatter plots, box plots, and feature importance analysis.
- **Machine Learning**: Predicts crop production using a pipeline with preprocessing and a RandomForestRegressor model.
- **Future Predictions**: Generates crop yield forecasts for years 2024–2030.
- **Performance Metrics**: Displays R² score, RMSE, MAE, and MAPE to evaluate the model.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/arshath4510/Predicting-Crop-Production-Based-on-Agricultural-Data.git
   cd Predicting-Crop-Production-Based-on-Agricultural-Data
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload a FAOSTAT CSV file through the app interface.
2. Analyze data cleaning steps and explore visualizations.
3. View model performance metrics and make predictions.
4. Check future production forecasts for selected crops and areas.

## Requirements
The project uses the following libraries (included in `requirements.txt`):
- pandas
- numpy
- scikit-learn
- streamlit
- plotly

To install all required libraries, run:
```bash
pip install -r requirements.txt
```

## Example CSV Format
The dataset should include the following columns:
- **Area**: Region or country name
- **Item**: Crop type
- **Element**: Data type (e.g., "Area harvested," "Yield," "Production")
- **Year**: Year of record
- **Value**: Corresponding numeric value

## Future Work
- Integration with external APIs for real-time weather and soil data.
- Support for additional machine learning models.
- Enhanced forecasting accuracy with time-series analysis.

---

Feel free to contribute or provide feedback to enhance the project!
