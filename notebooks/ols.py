import numpy as np
import pandas as pd
import statsmodels.api as sm

# Sample dataset with 25 observations
data = {
    'X1': list(range(1, 26)),  # X1 from 1 to 25
    'X2': [x * 2 for x in range(1, 26)],  # X2 = 2 * X1
    'Y': [
        1.1, 1.9, 3.0, 3.9, 5.1, 6.0, 7.1, 8.0, 9.2, 10.1,
        11.3, 12.2, 13.1, 14.0, 15.3, 16.2, 17.4, 18.5, 19.7, 20.9,
        21.2, 22.1, 23.4, 24.5, 25.6
    ]
}
df = pd.DataFrame(data)

# Independent variables (X) and dependent variable (Y)
X = df[['X1', 'X2']]
Y = df['Y']

# Add a constant (intercept) to the independent variables
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(Y, X).fit()

# Print the summary
print(model.summary())
