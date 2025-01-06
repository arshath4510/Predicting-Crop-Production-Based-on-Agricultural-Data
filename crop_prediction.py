import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class CropProductionPredictor:
    def __init__(self):
        self.le_area = LabelEncoder()
        self.le_item = LabelEncoder()
        self.feature_names = ['Area_encoded', 'Item_encoded', 'Year', 'Area_harvested', 'Yield']
        
        # Create preprocessing pipeline
        numeric_features = ['Year', 'Area_harvested', 'Yield']
        categorical_features = ['Area_encoded', 'Item_encoded']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler())  # RobustScaler handles outliers better
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        # Create model pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=200,  # Increased number of trees
                max_depth=20,      # Limited depth to prevent overfitting
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            ))
        ])
        
    def load_and_clean_data(self, data_path):
        """Load and clean the FAOSTAT dataset with improved preprocessing"""
        try:
            df = pd.read_csv(data_path)
            
            # Filter relevant columns
            cols = ['Area', 'Item', 'Element', 'Year', 'Value']
            df = df[cols]
            
            # Create separate dataframes for each element
            area_harvested = df[df['Element'] == 'Area harvested'].copy()
            yield_data = df[df['Element'] == 'Yield'].copy()
            production = df[df['Element'] == 'Production'].copy()
            
            # Merge the dataframes
            merged_df = pd.merge(
                area_harvested[['Area', 'Item', 'Year', 'Value']].rename(columns={'Value': 'Area_harvested'}),
                yield_data[['Area', 'Item', 'Year', 'Value']].rename(columns={'Value': 'Yield'}),
                on=['Area', 'Item', 'Year'],
                how='inner'
            )
            
            merged_df = pd.merge(
                merged_df,
                production[['Area', 'Item', 'Year', 'Value']].rename(columns={'Value': 'Production'}),
                on=['Area', 'Item', 'Year'],
                how='inner'
            )
            
            # Remove outliers using IQR method
            def remove_outliers(df, column):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
            # Apply outlier removal to numerical columns
            merged_df = remove_outliers(merged_df, 'Area_harvested')
            merged_df = remove_outliers(merged_df, 'Yield')
            merged_df = remove_outliers(merged_df, 'Production')
            
            # Log transform production values
            merged_df['Production_log'] = np.log1p(merged_df['Production'])
            
            # Remove rows with missing values
            merged_df = merged_df.dropna()
            
            if merged_df.empty:
                raise ValueError("No data remaining after preprocessing")
            
            return merged_df
            
        except Exception as e:
            st.error(f"Error in data loading and cleaning: {str(e)}")
            print(f"Error details: {str(e)}")
            return None
    
    def prepare_features(self, df):
        """Prepare features with improved preprocessing"""
        try:
            if df is None or df.empty:
                raise ValueError("Empty dataframe provided to prepare_features")
            
            # Encode categorical variables
            df['Area_encoded'] = self.le_area.fit_transform(df['Area'])
            df['Item_encoded'] = self.le_item.fit_transform(df['Item'])
            
            # Create feature matrix as DataFrame with named columns
            X = pd.DataFrame(df[self.feature_names])
            y = df['Production_log']  # Use log-transformed target
            
            return X, y
            
        except Exception as e:
            st.error(f"Error in feature preparation: {str(e)}")
            print(f"Error details: {str(e)}")
            return None, None
    
    def train_model(self, X, y):
        """Train the model with cross-validation"""
        try:
            if X is None or y is None:
                raise ValueError("Features or target is None")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_log = self.model.predict(X_test)
            
            # Transform predictions back to original scale
            y_pred = np.expm1(y_pred_log)
            y_test_original = np.expm1(y_test)
            
            # Calculate metrics on original scale
            metrics = {
                'R2': r2_score(y_test_original, y_pred),
                'MSE': mean_squared_error(y_test_original, y_pred),
                'MAE': mean_absolute_error(y_test_original, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test_original, y_pred)),
                'CV_scores': cross_val_score(self.model, X, y, cv=5)
            }
            
            # Calculate percentage errors
            metrics['MAPE'] = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
            
            return metrics
            
        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            print(f"Error details: {str(e)}")
            return None
    
    def create_streamlit_app(self):
        """Create Streamlit interface with improved visualizations"""
        try:
            st.title('Advanced Crop Production Prediction System')
            
            uploaded_file = st.file_uploader("Upload FAOSTAT CSV file", type=['csv'])
            
            if uploaded_file is not None:
                df = self.load_and_clean_data(uploaded_file)
                
                if df is not None and not df.empty:
                    X, y = self.prepare_features(df)
                    
                    if X is not None and y is not None:
                        metrics = self.train_model(X, y)
                        
                        if metrics is not None:
                            st.header('Model Performance')
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("R² Score", f"{metrics['R2']:.3f}")
                                st.metric("Root Mean Square Error", f"{metrics['RMSE']:.2f}")
                            
                            with col2:
                                st.metric("Mean Absolute Error", f"{metrics['MAE']:.2f}")
                               
                            
                            # Input parameters for prediction
                            st.sidebar.header('Input Parameters')
                            selected_area = st.sidebar.selectbox('Select Area', self.le_area.classes_)
                            selected_item = st.sidebar.selectbox('Select Crop', self.le_item.classes_)
                            selected_year = st.sidebar.slider('Select Year', 2000, 2030, 2024)
                            
                            # Use median values from similar crops/areas as default
                            default_area = df[df['Item'] == selected_item]['Area_harvested'].median()
                            default_yield = df[df['Item'] == selected_item]['Yield'].median()
                            
                            area_harvested = st.sidebar.number_input('Area Harvested (ha)', 
                                                                   min_value=0.0, 
                                                                   value=float(default_area))
                            yield_value = st.sidebar.number_input('Yield (kg/ha)', 
                                                                min_value=0.0,
                                                                value=float(default_yield))
                            
                            if st.sidebar.button('Predict'):
                                input_data = pd.DataFrame([[
                                    self.le_area.transform([selected_area])[0],
                                    self.le_item.transform([selected_item])[0],
                                    selected_year,
                                    area_harvested,
                                    yield_value
                                ]], columns=self.feature_names)
                                
                                # Make prediction (will be log-transformed)
                                prediction_log = self.model.predict(input_data)
                                prediction = np.expm1(prediction_log)[0]
                                
                                st.header('Predicted Production')
                                st.metric("Production", f"{prediction:.2f} tonnes")
                                
                                # Feature importance
                                if hasattr(self.model['regressor'], 'feature_importances_'):
                                    st.header('Feature Importance')
                                    importance_df = pd.DataFrame({
                                        'Feature': ['Area', 'Crop Type', 'Year', 'Area Harvested', 'Yield'],
                                        'Importance': self.model['regressor'].feature_importances_
                                    })
                                    fig = px.bar(importance_df, x='Feature', y='Importance',
                                               title='Feature Importance Analysis')
                                    st.plotly_chart(fig)
                                
                                # Show future predictions
                                if st.checkbox('Show Future Predictions (2024-2030)'):
                                    future_years = range(2024, 2031)
                                    future_predictions = []
                                    
                                    for year in future_years:
                                        future_input = input_data.copy()
                                        future_input['Year'] = year
                                        future_pred_log = self.model.predict(future_input)
                                        future_pred = np.expm1(future_pred_log)[0]
                                        future_predictions.append(future_pred)
                                    
                                    future_df = pd.DataFrame({
                                        'Year': future_years,
                                        'Predicted Production': future_predictions
                                    })
                                    
                                    fig = px.line(future_df, x='Year', y='Predicted Production',
                                                title=f'Future Production Predictions for {selected_item} in {selected_area}')
                                    st.plotly_chart(fig)
                            
                            # Data distribution plots
                            st.header('Data Analysis')
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.box(df, x='Item', y='Production',
                                           title='Production Distribution by Crop')
                                st.plotly_chart(fig)
                            
                            with col2:
                                fig = px.scatter(df, x='Area_harvested', y='Production',
                                               color='Item', title='Area vs Production')
                                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error in Streamlit app: {str(e)}")
            print(f"Error details: {str(e)}")

def main():
    predictor = CropProductionPredictor()
    predictor.create_streamlit_app()

if __name__ == "__main__":
    main()