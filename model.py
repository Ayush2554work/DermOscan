import tensorflow as tf
import numpy as np
from PIL import Image
import random
import os

# --- Configuration ---
MODEL_PATH = 'dermascan_model.h5'
IMG_WIDTH, IMG_HEIGHT = 224, 224
CLASS_NAMES = ['Benign', 'Malignant']

# --- Load the Trained Model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Error: Model file not found at {MODEL_PATH}. "
                            "Did you run the 'train_model.py' script first?")
                            
model = tf.keras.models.load_model(MODEL_PATH)
print("TensorFlow model loaded successfully.")

def _preprocess_image(image_path):
    """Loads and preprocesses an image for the model."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale to [0, 1]
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    return img_array

def get_real_prediction(image_path):
    """
    Performs a real prediction on the user's image.
    """
    # 1. Preprocess the image
    processed_image = _preprocess_image(image_path)
    
    # 2. Get prediction from the model
    # The output is a probability (0.0 to 1.0) due to the sigmoid function
    prediction_prob = model.predict(processed_image)[0][0]
    
    # 3. Determine class
    # We set a 0.5 threshold.
    if prediction_prob >= 0.5:
        prediction_index = 1 # Malignant
        confidence = prediction_prob
    else:
        prediction_index = 0 # Benign
        confidence = 1 - prediction_prob
        
    prediction_class = CLASS_NAMES[prediction_index]
    confidence_percent = f"{confidence * 100:.1f}%"
    
    print(f"Prediction: {prediction_class}, Confidence: {confidence_percent}")
    
    # 4. Build the report dictionary
    # This is a basic report. You can expand this.
    report_data = {
        "image_path": image_path,
        "report_id": f"DMS-{random.randint(1000, 9999)}",
        "prediction": prediction_class,
        "confidence": confidence_percent,
    }
    
    if prediction_class == 'Malignant':
        report_data["status"] = "Warning"
        report_data["factors"] = {
            "Suspicion Level": "High",
            "Analysis": "The AI model detected patterns consistent with malignant lesions.",
            "Asymmetry": "Likely",
            "Border": "Potentially irregular",
            "Color": "Potentially non-uniform"
        }
        report_data["recommendation"] = ("Urgent consultation with a certified dermatologist is "
                                         "highly recommended for a professional biopsy and diagnosis.")
    else:
        report_data["status"] = "Clear"
        report_data["factors"] = {
            "Suspicion Level": "Low",
            "Analysis": "The AI model detected patterns consistent with benign lesions.",
            "Asymmetry": "Likely symmetrical",
            "Border": "Likely regular",
            "Color": "Likely uniform"
        }
        report_data["recommendation"] = ("Lesion appears benign. Continue to monitor for any changes "
                                         "as part of a regular self-examination. Consult a doctor "
                                         "if you notice any changes.")
    
    return report_data