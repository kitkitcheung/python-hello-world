import pandas as pd

# Create a dictionary of data
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
        'Age': [25, 32, 18, 47, 29],
        'City': ['New York', 'Paris', 'London', 'San Francisco', 'Berlin']}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

# Select rows where Age is greater than 30
df_above_30 = df[df['Age'] > 30]

# Print the new DataFrame
print(df_above_30)

# Calculate the mean age
mean_age = df['Age'].mean()

# Print the mean age
print(mean_age)
