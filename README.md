### Project Overview
- **Objective**: Predict life expectancy for a country based on input features like income, schooling, health metrics, and economic indicators using a trained Random Forest Regression model.
- **Dataset**: The project uses the "Life Expectancy Data.csv" dataset, which contains country-level data on factors influencing life expectancy.
- **Tools and Libraries**:
  - **Data Processing and Modeling**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
  - **Deployment**: `streamlit`, `pickle` (for model serialization)
- **Components**:
  - **Jupyter Notebook (`workshop_22_4.ipynb`)**: Handles data exploration, preprocessing, and model training.
  - **Streamlit App (`deploy.py`)**: Provides a user-friendly interface for inputting feature values and obtaining life expectancy predictions.

---

### 1. Jupyter Notebook (`workshop_22_4.ipynb`)

#### Purpose
The notebook is used for:
- Exploring and preprocessing the "Life Expectancy Data.csv" dataset.
- Training a Random Forest Regression model.
- Visualizing model performance.

#### Key Steps
1. **Data Loading and Exploration**:
   - **Dataset**: Loaded using `pandas.read_csv("Life Expectancy Data.csv")`.
   - **Structure**: The dataset has 2,938 entries and 22 columns, including:
     - **Target**: `Life expectancy`
     - **Features**: `Country`, `Year`, `Status`, `Adult Mortality`, `infant deaths`, `Alcohol`, `percentage expenditure`, `Hepatitis B`, `Measles`, `BMI`, `under-five deaths`, `Polio`, `Total expenditure`, `Diphtheria`, `HIV/AIDS`, `GDP`, `Population`, `thinness 1-19 years`, `thinness 5-9 years`, `Income composition of resources`, `Schooling`
   - **Missing Values**: Identified using `df.isnull().sum()`. Columns with missing values include `Life expectancy` (10), `Alcohol` (194), `Hepatitis B` (553), `Total expenditure` (226), `GDP` (448), `Population` (652), `Income composition of resources` (167), and `Schooling` (163).
   - **Data Types**: Mostly numerical (`float64`, `int64`), with `Country` and `Status` as categorical (`object`).

2. **Data Insights**:
   - **Summary Statistics**: Generated using `df.describe()`, showing ranges and distributions of features (e.g., `Life expectancy` ranges from 36.3 to 89.0 years, mean 69.22).
   - **Feature Analysis**: Features like `Income composition of resources`, `Schooling`, and `Adult Mortality` are likely key predictors based on their inclusion in the model.

3. **Model Training** (Inferred from Context):
   - **Feature Selection**: The model uses the top 15 correlated features, as listed in `deploy.py`:
     - `Income composition of resources`, `Schooling`, `Adult Mortality`, `HIV/AIDS`, `BMI`, `thinness 1-19 years`, `thinness 5-9 years`, `Diphtheria`, `Polio`, `Total expenditure`, `GDP`, `Hepatitis B`, `Alcohol`, `Status_Developed`, `under-five deaths`
   - **Preprocessing**:
     - **Categorical Encoding**: The `Status` column (Developing/Developed) is encoded as `Status_Developed` (binary: 0 for Developing, 1 for Developed).
     - **Scaling**: Features are standardized using `StandardScaler` (saved as `scaler.pkl`).
   - **Model**: A Random Forest Regression model is trained and saved as `rf_model.pkl`.
   - **Evaluation**: A scatter plot of actual vs. predicted life expectancy is generated, indicating model performance (though metrics like MSE or R² are not shown in the provided snippet).

4. **Visualization**:
   - A scatter plot (`plt.scatter(y_test, y_pred)`) visualizes actual vs. predicted life expectancy, helping assess model accuracy.

#### Outputs
- **Model Files**: `rf_model.pkl` (trained Random Forest model) and `scaler.pkl` (fitted StandardScaler).
- **Insights**: The dataset has significant missing values, requiring imputation or handling during preprocessing. The selected features suggest strong correlations with life expectancy.

---

### 2. Streamlit Application (`deploy.py`)

#### Purpose
The Streamlit app provides an interactive interface for users to input feature values and predict life expectancy using the pre-trained Random Forest model.

#### Key Features
1. **Model and Scaler Loading**:
   - Loads `rf_model.pkl` and `scaler.pkl` from `C:/Users/Mariam/Downloads/`.
   - Includes error handling for missing files, displaying an error message if files are not found.

2. **User Interface**:
   - **Title**: "Life Expectancy Prediction"
   - **Input Form**: A form collects values for the 15 top features:
     - **Numeric Inputs**: Most features (e.g., `Income composition of resources`, `Schooling`, `Adult Mortality`) use `st.number_input` with a step of 0.1 and two-decimal precision.
     - **Binary Input**: `Status_Developed` uses a dropdown (`st.selectbox`) with options 0 (Developing) and 1 (Developed).
   - **Submit Button**: Triggers the prediction process.

3. **Prediction Process**:
   - **Input Processing**: User inputs are collected into a dictionary and converted to a `pandas.DataFrame` with the exact feature order.
   - **Scaling**: The input data is scaled using the loaded `StandardScaler`.
   - **Prediction**: The scaled data is passed to the Random Forest model to predict life expectancy.
   - **Output**: Displays the predicted life expectancy (e.g., "Predicted Life Expectancy: 72.50 years") and the input values in JSON format for verification.
   - **Error Handling**: Catches and displays any errors during prediction.

4. **Additional Information**:
   - A markdown section explains the model, noting it’s a Random Forest Regression model trained on the Life Expectancy dataset using the top 15 correlated features.

#### Dependencies
- `streamlit`, `pandas`, `numpy`, `pickle`, `sklearn.preprocessing.StandardScaler`

### Project Workflow
1. **Data Preparation** (Notebook):
   - Load and clean the dataset (handle missing values, encode `Status`).
   - Select top 15 correlated features.
   - Scale features and train a Random Forest model.
   - Save the model and scaler as `rf_model.pkl` and `scaler.pkl`.

2. **Model Deployment** (Streamlit App):
   - Load the model and scaler.
   - Collect user inputs via a form.
   - Scale inputs and predict life expectancy.
   - Display results and input verification.

3. **User Interaction**:
   - Users enter values for features like `Schooling`, `HIV/AIDS`, etc.
   - The app outputs a predicted life expectancy and shows input values for transparency.

---

### Strengths
- **Interactive Interface**: The Streamlit app makes the model accessible to non-technical users.
- **Robust Model**: Random Forest is effective for capturing complex relationships in the data.
- **Feature Selection**: Using the top 15 correlated features reduces dimensionality while maintaining predictive power.
- **Error Handling**: The app includes checks for missing model files and prediction errors.

---

### Conclusion
The Life Expectancy Prediction project is a well-structured machine learning application that combines data analysis, model training, and deployment. The Jupyter Notebook provides a foundation for exploring and modeling the dataset, while the Streamlit app offers an intuitive interface for predictions. The primary challenge is the `urllib3` compatibility issue due to Python 3.7’s outdated OpenSSL. By following the recommended fix (using a virtual environment and specific package versions) or upgrading Python, the app should run successfully. With minor enhancements, the project could be production-ready and more robust.
