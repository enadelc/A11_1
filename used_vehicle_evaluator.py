# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

# Load dataset
data = pd.read_csv('data/vehicles.csv')
# Remove NaN values
data.dropna(inplace=True)
# Initial data inspection
print(data.head())
print(data.info())
print(data.describe(include='all'))

# Data Cleaning
data.drop_duplicates(inplace=True)
data.dropna(subset=['price', 'year', 'manufacturer', 'model', 'condition', 'odometer'], inplace=True)  # Drop rows with essential missing values

# Feature Engineering
data['age'] = 2024 - data['year']
data['mileage_per_year'] = data['odometer'] / data['age']

# Define features and target variable
X = data[['age', 'mileage_per_year', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'drive', 'size', 'type', 'paint_color', 'state', 'transmission']]
y = data['price']

# Data Transformation
numeric_features = ['age', 'mileage_per_year']
categorical_features = ['manufacturer', 'model', 'condition', 'fuel', 'drive', 'size', 'type', 'paint_color', 'state', 'transmission', 'cylinders']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define models
models = {
    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', LinearRegression())]),  # Set the number of features to 8
    'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', GradientBoostingRegressor(n_estimators=512, max_depth=100,max_features=8))]),  # Set the number of trees to 100 and max depth to 3    
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', RandomForestRegressor(n_estimators=512, max_depth=100,max_features=8))])  # Set the number of trees to 512 and max depth to 10}
}
# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'R2': r2}
    
    print(f"{name} - MSE: {mse}, R2: {r2}")

# Model Comparison Plot

results_df = pd.DataFrame(results).T

# Plot R2 on the right axis
ax = results_df.plot(kind='bar', figsize=(10, 6))
ax2 = ax.twinx()
ax2.plot(results_df.index, results_df['R2'], color='red', marker='o')
ax2.set_ylabel('R2')
plt.title('Model Comparison')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print top five important features for each model
for name, model in models.items():
    if name == 'Linear Regression':
        feature_importances = model.named_steps['regressor'].coef_
        feature_names = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
        top_features = pd.Series(feature_importances, index=numeric_features + list(feature_names))
        top_features = top_features.abs().sort_values(ascending=False).head(5)
        print(f"Top five important features for {name}:")
        print(top_features)
        print()
    elif name == 'K-Nearest Neighbors':
        print(f"Top five important features for {name}:")
        print("K-Nearest Neighbors does not provide feature importances")
        print()
    else:
        feature_importances = model.named_steps['regressor'].feature_importances_
        feature_names = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
        top_features = pd.Series(feature_importances, index=numeric_features + list(feature_names))
        top_features = top_features.abs().sort_values(ascending=False).head(5)
        print(f"Top five important features for {name}:")
        print(top_features)
        print()
