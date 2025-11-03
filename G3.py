import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'width': [640, 800, 720, 1024, 640],
    'height': [480, 600, 720, 768, 480],
    'file_size_kb': [120, 200, 150, 300, 110],
    'category': ['cat', 'dog', 'dog', 'bird', 'cat']
}
df = pd.DataFrame(data)

# 🎯 1. Histogram – distribution of image widths
# plt.figure(figsize=(5,3))
plt.hist(df['width'], bins=5, edgecolor='black')
plt.title('Image Width Distribution')
plt.xlabel('Width (px)')
plt.ylabel('Frequency')
plt.show()

# 🎯 2. Scatter Plot – relationship between width and height
# plt.figure(figsize=(5,3))
sns.scatterplot(x='width', y='height', hue='category', data=df, s=100)
plt.title('Width vs Height by Category')
plt.show()

# 🎯 3. Correlation Heatmap – show numerical relationships
# plt.figure(figsize=(4,3))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title('Correlation Heatmap')
plt.show()
