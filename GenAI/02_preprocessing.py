import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', np.nan, 'Eve'],
    'Age': [25, np.nan, 30, 22, 28],
    'Salary': [50000, 60000, np.nan, 45000, 52000],
    'Department': ['HR', 'IT', 'Finance', 'IT', np.nan]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# Handling missing values
# Fill numerical NaN with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)
# Fill categorical NaN with mode
df['Name'].fillna(df['Name'].mode()[0], inplace=True)
df['Department'].fillna(df['Department'].mode()[0], inplace=True)

print("\nDataset after Handling Missing Values:\n", df)

# Normalizing numerical features
scaler = MinMaxScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])
print("\nDataset after Normalization:\n", df)

# Encoding categorical variables
# Label Encoding for Department
le = LabelEncoder()
df['Department_Encoded'] = le.fit_transform(df['Department'])

# One-Hot Encoding for Department
df_onehot = pd.get_dummies(df, columns=['Department'])

print("\nDataset with Label Encoding:\n", df)
print("\nDataset with One-Hot Encoding:\n", df_onehot)
