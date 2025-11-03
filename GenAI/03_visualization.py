import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Sample dataset for visualization
data = {
'ImageID': [1,2,3,4,5,6,7,8,9,10],
'Width': [1024, 800, 640, 1024, 800, 640, 1200, 1024, 640, 800],
'Height': [768, 600, 480, 768, 600, 480, 900, 768, 480, 600],
'Brightness': [0.8,0.6,0.75,0.82,0.65,0.78,0.9,0.81,0.74,0.68],
'ObjectCount': [3,1,2,4,2,3,5,4,1,2],
'Category': ['Cat','Dog','Dog','Cat','Bird','Bird','Dog','Cat','Bird','Dog']
}
df = pd.DataFrame(data)

# Histogram of Brightness
plt.figure(figsize=(6,4))
sns.histplot(df['Brightness'], bins=5, kde=True)
plt.title('Brightness Distribution')
plt.xlabel('Brightness')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of Width vs Height
plt.figure(figsize=(6,4))
sns.scatterplot(x='Width', y='Height', hue='Category', data=df, s=100)
plt.title('Width vs Height by Category')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.show()
# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df[['Width','Height','Brightness','ObjectCount']].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
# Box plot of ObjectCount by Category
plt.figure(figsize=(6,4))
sns.boxplot(x='Category', y='ObjectCount', data=df)
plt.title('Object Count by Category')
plt.show()
