DermOscan - AI-Driven Skin Cancer Detection

[ Application Developed by AYUSH ]

DermOscan is a desktop application that utilizes advanced computer vision and deep learning to provide a preliminary analysis of skin lesion images. It is designed to demonstrate the potential of AI in assisting early detection of skin irregularities.

Disclaimer: This application is for educational and informational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for any health concerns.

✨ Features

AI-Powered Analysis: Uses a custom-trained MobileNetV2 model to classify lesions as 'Benign' or 'Malignant' (Warning).

Modern UI: Built with CustomTkinter for a sleek, dark-mode user interface.

Live Camera Capture: Integrated webcam support to take photos directly within the app.

Image Upload: Support for uploading existing images (JPG, PNG).

Instant Reports: Generates a detailed on-screen report with confidence scores and key analysis factors.

PDF Download: Export reports to PDF format for easy sharing or record-keeping.

Skin Detection Check: Automatically validates images to ensure they contain visible skin before analysis.

🛠️ Tech Stack

Language: Python 3.x

GUI: CustomTkinter

AI/ML: TensorFlow, Keras (MobileNetV2), OpenCV, NumPy

Imaging: Pillow (PIL)

Reporting: ReportLab (PDF generation)

🚀 Installation & Setup

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/DermOscan.git](https://github.com/YOUR_USERNAME/DermOscan.git)
cd DermOscan


Create a virtual environment (Recommended):

# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Required Assets:
Ensure the following files are in the root directory:

dermascan_model.h5 (The trained AI model)

logo.png (App logo for home screen)

logo.ico (App icon for window title bar)

DOS1.jpg to DOS6.jpg (Slideshow images)

Run the application:

python app.py


🧠 Model Training (Optional)

The model was trained on the HAM10000 dataset from Kaggle.
To retrain the model yourself:

Download the dataset into the project folder.

Run the training script:

python train_model.py


This will generate a new dermascan_model.h5 file.

📄 License

This project is open-source and available under the MIT License.