
Used Car Price Analysis
Summary of Findings
In this project, we aimed to identify the key drivers affecting used car prices. We analyzed a dataset containing various features such as vehicle age, mileage, make, model, and condition. Our analysis revealed that factors like vehicle age and mileage are significant predictors of price. We built and compared multiple regression models, including Linear Regression, Random Forest, and Gradient Boosting, optimizing hyperparameters and validating model performance through cross-validation. The final model provides valuable insights for pricing strategies and inventory management.

Key Insights
Significant Features: Vehicle age and mileage are crucial drivers of used car prices.
Model Performance: The selected model demonstrates strong predictive accuracy and reliable performance metrics.
Recommendations: Adjust pricing based on vehicle age and mileage; consider optimizing inventory based on these insights.
Notebook
For detailed analysis, methodology, and results, please refer to the prompt_II where the models and findings are elaborated.

Snippet of Results:

Linear Regression - MSE: 92038296.64641926, R2: 0.5127299385145626
Gradient Boosting - MSE: 45754142.68705401, R2: 0.7577679647203505
Random Forest - MSE: 89977228.91472253, R2: 0.5236416637087793
Top five important features for Linear Regression:
model_benz sprinter           114278.431500
model_challenger srt demon    101050.294715
model_5 window coupe           88114.664943
model_850i                     84044.609071
model_hot rod                  83465.557273
dtype: float64

Top five important features for Gradient Boosting:
age                      0.132557
mileage_per_year         0.069720
cylinders_8 cylinders    0.030959
cylinders_8 cylinders    0.030959
fuel_diesel              0.030813
type_truck               0.029214
dtype: float64
type_truck               0.029214
dtype: float64
dtype: float64

Top five important features for Random Forest:
age                 0.098791
mileage_per_year    0.061506
fuel_diesel         0.044064
fuel_gas            0.041248
type_truck          0.032176
dtype: float64

# A11_1
