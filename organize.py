import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# 1. Load the labels
df = pd.read_csv('images.csv')

# 2. Filter for the top 10 categories (as per assignment requirements)
# The subset repo is already small, but let's ensure we have the right classes
top_10 = df['label'].value_counts().nlargest(10).index
df_subset = df[df['label'].isin(top_10)].sample(n=1000, random_state=42)

# 3. Split the data: 70% train, 15% val, 15% test
train, test = train_test_split(df_subset, test_size=0.3, stratify=df_subset['label'], random_state=42)
val, test = train_test_split(test, test_size=0.5, stratify=test['label'], random_state=42)

# 4. Function to move files into a directory structure
def organize_files(data, split_name):
    for _, row in data.iterrows():
        label = row['label']
        img_name = f"{row['image']}.jpg"
        source = os.path.join('images', img_name)
        target_dir = os.path.join('dataset', split_name, label)
        
        os.makedirs(target_dir, exist_ok=True)
        if os.path.exists(source):
            shutil.copy(source, os.path.join(target_dir, img_name))

# Execute the move
organize_files(train, 'train')
organize_files(val, 'validation')
organize_files(test, 'test')

print("Data organized successfully!")