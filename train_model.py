import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10 # Increase for better accuracy, e.g., 25
MODEL_SAVE_PATH = 'dermascan_model.h5'

print("Step 1: Loading Metadata...")
# Load the metadata
base_dir = '.' # Assumes script is in the same folder
metadata_path = os.path.join(base_dir, 'HAM10000_metadata.csv')
df = pd.read_csv(metadata_path)

# --- Create image path dictionary ---
# This dataset is split into two folders. We'll build a map.
image_path_map = {}
for folder in ['ham10000_images_part_1', 'ham10000_images_part_2']:
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found - {folder_path}")
        print("Please make sure the Kaggle image folders are in the same directory.")
        exit()
        
    for img_file in os.listdir(folder_path):
        image_id = os.path.splitext(img_file)[0]
        image_path_map[image_id] = os.path.join(folder_path, img_file)

# Add a 'path' column to our dataframe
df['path'] = df['image_id'].map(image_path_map)

# Drop any rows where the image path couldn't be found
df = df.dropna(subset=['path'])

print(f"Found {df.shape[0]} images.")

# --- Simplification: 7 classes to 2 (Malignant vs. Benign) ---
# This matches your app's desired output
# 'akiec', 'bcc', 'mel' are generally considered malignant or pre-malignant
# 'nv' (moles), 'bkl', 'df', 'vasc' are generally benign
# 'scc' (Squamous cell carcinoma) is also malignant
malignant_classes = ['akiec', 'bcc', 'mel', 'scc']
df['binary_label'] = df['dx'].apply(lambda x: 1 if x in malignant_classes else 0)

# --- Label Encoding ---
# 0 = Benign, 1 = Malignant
class_names = ['Benign', 'Malignant']
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['binary_label'])

print("Step 2: Splitting Data...")
# Split the data
X = df['path']
y = df['label_encoded']
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42, stratify=y)
y_train = df.loc[X_train.index]['label_encoded']
y_val = df.loc[X_val.index]['label_encoded']

# Create dataframes for the generators
train_df = pd.DataFrame({'path': X_train, 'label': y_train.astype(str)})
val_df = pd.DataFrame({'path': X_val, 'label': y_val.astype(str)})

print(f"Training data: {len(train_df)} samples")
print(f"Validation data: {len(val_df)} samples")

# --- Image Data Generators ---
print("Step 3: Setting up Image Generators...")
# Use data augmentation for the training set to prevent overfitting
train_datgen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255) # No augmentation for validation

train_generator = train_datgen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary' # We are doing binary classification
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='label',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --- Build the Model ---
print("Step 4: Building MobileNetV2 Model...")
# Load the pre-trained MobileNetV2 model without its top classification layer
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)

# Freeze the base model layers
base_model.trainable = False

# Add our custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) # Add dropout for regularization
x = Dense(128, activation='relu')(x)
# The final layer has 1 neuron and a sigmoid activation for binary classification
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- Compile the Model ---
print("Step 5: Compiling Model...")
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- Train the Model ---
print(f"Step 6: Starting Training for {EPOCHS} epochs...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# --- Save the Model ---
print(f"Step 7: Training complete. Saving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("--- Model Saved Successfully! You can now run the app.py ---")