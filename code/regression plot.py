import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample DataFrame (replace with your actual DataFrame)
data = {
    'Mar Cap - Crore': [583436.72, 563709.84, 482953.59, 320985.27, 289497.37, 250000],
    'Sales Qtr - Crore': [99810.00, 30904.00, 20581.27, 9772.02, 16840.51, 15000]
}
df = pd.DataFrame(data)

# Create a linear regression model
X = df[['Mar Cap - Crore']]
y = df['Sales Qtr - Crore']
model = LinearRegression()
model.fit(X, y)
df['Predicted Sales'] = model.predict(X)

# Create the scatter plot with a regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='Mar Cap - Crore', y='Sales Qtr - Crore', data=df, scatter_kws={'s':50}, line_kws={'color':'red'})
plt.title('Market Capitalization vs Quarterly Sales')
plt.xlabel('Market Capitalization in Crores')
plt.ylabel('Quarterly Sales in Crores')
plt.grid(True)

# Save the plot as an image
plt.savefig('regression_plot.png')

plt.show()
