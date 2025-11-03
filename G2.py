import pandas as pd
import numpy as np

# Sample dataset (e.g., image metadata)
data = {
    'image_id': [1, 2, 3, 4, 5],
    'caption': ['cat on mat', np.nan, 'dog in park', 'bird flying', 'cat sleeping'],
    'width': [640, 800, np.nan, 1024, 640],
    'height': [480, 600, 720, np.nan, 480],
    'category': ['animal', 'animal', 'animal', 'bird', np.nan]
}

df = pd.DataFrame(data)
print("Original Data:\n", df, "\n")

# Handle missing data
df['caption'].fillna('unknown', inplace=True)
df['width'].fillna(df['width'].mean(), inplace=True)
df['height'].fillna(df['height'].mean(), inplace=True)
df['category'].fillna(df['category'].mode()[0], inplace=True)

# Normalize numerical features
df['width'] = (df['width'] - df['width'].min()) / (df['width'].max() - df['width'].min())
df['height'] = (df['height'] - df['height'].min()) / (df['height'].max() - df['height'].min())

# Encode categorical variables
df = pd.get_dummies(df, columns=['category'], prefix='cat')

print("Cleaned and Preprocessed Data:\n", df)