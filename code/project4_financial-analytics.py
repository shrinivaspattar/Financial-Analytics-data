# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go


# Function to load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Function to explore the dataset
def explore_data(df):
    print("First few rows of the dataset:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nMissing values in the dataset:")
    print(df.isnull().sum())


# Function to clean and preprocess the data
def preprocess_data(df):
    # Drop rows with missing values in specific columns
    df = df.dropna(subset=['Mar Cap - Crore', 'Sales Qtr - Crore']).copy()

    # Convert data types
    df['Mar Cap - Crore'] = df['Mar Cap - Crore'].astype(float)
    df['Sales Qtr - Crore'] = df['Sales Qtr - Crore'].astype(float)

    return df


# Function to calculate key metrics
def calculate_metrics(df):
    market_cap_mean = df['Mar Cap - Crore'].mean()
    sales_qtr_mean = df['Sales Qtr - Crore'].mean()
    market_cap_median = df['Mar Cap - Crore'].median()
    sales_qtr_median = df['Sales Qtr - Crore'].median()

    print(f'Average Market Capitalization: {market_cap_mean:.2f}')
    print(f'Average Quarterly Sales: {sales_qtr_mean:.2f}')
    print(f'Median Market Capitalization: {market_cap_median:.2f}')
    print(f'Median Quarterly Sales: {sales_qtr_median:.2f}')

    return market_cap_mean, sales_qtr_mean, market_cap_median, sales_qtr_median


# Function to visualize the data
def visualize_data(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Mar Cap - Crore', y='Sales Qtr - Crore', data=df)
    plt.title('Market Capitalization vs Quarterly Sales')
    plt.xlabel('Market Capitalization in Crores')
    plt.ylabel('Quarterly Sale in crores')

    # Format the x and y axis labels
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

    # Save the plot as an image file
    plt.savefig('market_cap_vs_sales.png')
    plt.show()


# Function to calculate descriptive statistics
def descriptive_statistics(df):
    desc_stats = df.describe()
    print("\nDescriptive Statistics:")
    print(desc_stats)
    return desc_stats


# Function to calculate and visualize correlation
def correlation_analysis(df):
    # Select only numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include='number')
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    return correlation_matrix


# Function to visualize the distribution of key attributes
def visualize_distributions(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Mar Cap - Crore'], bins=30, kde=True)
    plt.title('Distribution of Market Capitalization')
    plt.xlabel('Market Capitalization in Crores')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['Sales Qtr - Crore'], bins=30, kde=True)
    plt.title('Distribution of Quarterly Sales')
    plt.xlabel('Quarterly Sales in Crores')
    plt.ylabel('Frequency')
    plt.show()


# Function to perform regression analysis
def regression_analysis(df):
    X = df[['Mar Cap - Crore']]
    y = df['Sales Qtr - Crore']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
    plt.title('Regression Analysis: Market Cap vs Quarterly Sales')
    plt.xlabel('Market Capitalization in Crores')
    plt.ylabel('Quarterly Sales in Crores')
    plt.legend()
    plt.show()


# Function to create interactive visualizations using Plotly
def interactive_visualizations(df):
    fig = px.scatter(df, x='Mar Cap - Crore', y='Sales Qtr - Crore', title='Market Capitalization vs Quarterly Sales')
    fig.update_layout(xaxis_title='Market Capitalization in Crores', yaxis_title='Quarterly Sale in crores')
    fig.show()


# Main function to execute the project steps
def main():
    # Define the file path
    file_path = r"D:\internship\Unified Mentor Internship project\my project\project 4\Financial Analytics data.csv"

    # Load the dataset
    df = load_data(file_path)

    # Explore the dataset
    explore_data(df)

    # Clean and preprocess the data
    df = preprocess_data(df)

    # Calculate key metrics
    market_cap_mean, sales_qtr_mean, market_cap_median, sales_qtr_median = calculate_metrics(df)

    # Descriptive statistics
    descriptive_statistics(df)

    # Correlation analysis
    correlation_analysis(df)

    # Visualize distributions
    visualize_distributions(df)

    # Visualize the data
    visualize_data(df)

    # Perform regression analysis
    regression_analysis(df)

    # Create interactive visualizations
    interactive_visualizations(df)


# Execute the main function
if __name__ == "__main__":
    main()
