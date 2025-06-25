import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import json

# ====== Configurations ======
CSV_PATH = "data/labels.csv"
IMG_FOLDER = "data/images"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "flask/model.h5"
CLASS_MAP_PATH = "flask/class_map.json"

# ====== Load Dataset ======
df = pd.read_csv(CSV_PATH)
print("CSV Columns:", df.columns)

# Convert labels to indices
class_names = sorted(df['class'].unique())
class_to_index = {name: i for i, name in enumerate(class_names)}
print("Class to Index Map:", class_to_index)

# Save class mapping
os.makedirs(os.path.dirname(CLASS_MAP_PATH), exist_ok=True)
with open(CLASS_MAP_PATH, 'w') as f:
    json.dump(class_names, f)

# ====== Load and Preprocess Images ======
X = []
y = []

print("Loading images...")
for i in tqdm(range(len(df))):
    # Build full image path
    filename = df.iloc[i]['filename']
    label = df.iloc[i]['class']
    img_path = os.path.join(IMG_FOLDER, filename)

    # Load and preprocess image
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0

    X.append(img)
    y.append(class_to_index[label])

X = np.array(X)
y = to_categorical(y, num_classes=len(class_names))

# ====== Train/Test Split ======
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== Define CNN Model ======
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ====== Train Model ======
print("Training model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# ====== Save Model ======
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
