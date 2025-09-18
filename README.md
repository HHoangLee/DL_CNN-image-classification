# DL_CNN-image-classification
This project implements a simple Convolutional Neural Network (CNN) for classifying animal images using PyTorch. It is trained on three classes: cat, dog, butterfly.

🐾 CNN Image Classification (Cats, Dogs, Butterflies)
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify animal images into 3 categories: cat, dog, and butterfly.
It was developed and trained on Google Colaboratory to leverage free GPU resources.

🚀 Features
Preprocessing pipeline with torchvision.transforms (resize, tensor conversion).
Simple CNN model with 2 convolutional layers + fully connected layers.
Training with CrossEntropyLoss and Adam optimizer.
Performance evaluation using accuracy_score & classification_report.
Save & load trained models for reuse.
Custom function to predict new images.

📂 Dataset
Dataset includes multiple animal classes (cat, dog, butterfly, cow, etc.) but only 3 classes were used: cat, dog, butterfly.
Each class is stored in .zip format and extracted before training.
Directory structure after preprocessing:
dataset/
│── cat/
│── dog/
│── butterfly/

⚙️ Workflow
1.Extract dataset from .zip files into folders.
2.Preprocess images: resize → tensor conversion.
3.Split dataset into train (80%) & validation (20%).
4.Define CNN model:
   2 Conv2D + ReLU + MaxPooling layers
   Flatten → Fully connected → Softmax
5.Train model with 5 epochs.
6.Evaluate performance with accuracy & classification report.
7.Predict new images with predict_image().
8.Save & reload model to avoid retraining.

🛠️ Technologies
Google Colab (GPU support)
Python libraries:
  PyTorch, torchvision
  sklearn
  matplotlib
  PIL
  zipfile, shutil, os

📊 Example Results
Accuracy: ~80–90% (depends on dataset size & epochs).

🔮 Future Improvements
Use more classes (multi-class classification).
Apply data augmentation (rotation, flip, normalization).
Experiment with deeper CNNs (ResNet, VGG).
Deploy as a web app (Flask/Django + frontend).

📌 How to Run
# Clone repo
git clone https://github.com/your-username/cnn-animal-classification.git
cd cnn-animal-classification
# Run notebook in Google Colab or Jupyter

👉 With this setup, you can easily train, evaluate, and predict animal images using CNN.
