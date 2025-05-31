# Two-Stream CNN with Transformer Attention for PCOS Image Classification
### Download :- [Dataset(PCOS)](https://drive.google.com/drive/folders/1wW_mjlxlU2MMqjJlICzA1Xn8fyAYqeFd?usp=sharing)
## 📄 Description

This project focuses on classifying Polycystic Ovary Syndrome (PCOS) ultrasound images into two categories: "infected" and "noninfected." The model employs a two-stream Convolutional Neural Network (CNN) combined with a Transformer attention mechanism to enhance feature extraction and classification accuracy. The dataset consists of 11,784 images, which were preprocessed, balanced, and split into training, validation, and test sets. The model achieved a test accuracy of 99.34%, demonstrating its effectiveness in PCOS diagnosis.

## 🚀 Features

- 📁 Handles over 11,000 labeled ultrasound images from the PCOS-XAI dataset.
- 🔍 Data balancing via upsampling for handling class imbalance.
- 🧠 Custom two-stream CNN architecture to process top and bottom halves of images.
- 🧲 Incorporates **Transformer-based Multi-Head Attention** for effective feature fusion.
- 🖼️ Augmented training pipeline using TensorFlow’s `ImageDataGenerator`.
- 📊 Visual analytics for dataset distribution and model performance.
- 📈 Achieves over **99% test accuracy** on balanced, stratified data splits.

## 🛠 Installation Instructions

1. Clone the repository or download the project files.
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
3. Required libraries include:
- TensorFlow
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- OpenCV
- PIL

## 📊 Reports
🔹 Model Performance

Training Accuracy: 98.85%

Validation Accuracy: 98.82%

Test Accuracy: 99.34%

Test Loss: 0.0202

🔹 Classification Report
```text
                 precision    recall  f1-score   support

       0           0.99        0.99      0.99       679
       1           0.99        0.99      0.99       678

    accuracy                             0.99      1357
   macro avg       0.99        0.99      0.99      1357
weighted avg       0.99        0.99      0.99      1357
```
🔹 Confusion Matrix
```text
                     Predicted: Infected	Predicted: Non-Infected
Actual: Infected	     670	                 9
Actual: Non-Infected	      8	                        670
```
(Exact values may vary slightly based on random seed and training session)

![Confusion Matrix](https://github.com/Harsha-096/PCOS-Ultrasound-Image-Classification-using-Two-Stream-CNN-Transformer-Attention/blob/6630b41fe7be834d6dacd094057a41faf03f0256/Reports/Confusion%20Matrix.png)

🔹 Dataset Summary

Total Samples: 11,784

Infected: 6,784

Non-Infected: 5,000

After Balancing: 6,784 each

![Distribution of PCOS Types - Pie Chart](https://github.com/Harsha-096/PCOS-Ultrasound-Image-Classification-using-Two-Stream-CNN-Transformer-Attention/blob/6630b41fe7be834d6dacd094057a41faf03f0256/Reports/Distribution%20of%20PCOS%20Types%20-%20Pie%20Chart.png)

## ▶️ Usage

1. Prepare Dataset:

Ensure the dataset is structured as:
```text
/PCOS/
  ├── infected/
  └── noninfected/
```
2. Run Preprocessing:

Processes and balances the data using label encoding and upsampling.

3. Train the Model:
```text
model.fit(train_gen_new, validation_data=valid_gen_new, epochs=3)
```
4. Evaluate:
   
Use the evaluation script to generate confusion matrix and classification report.

6. Visualize:
   
Plot training history and heatmaps to assess model learning and accuracy.

## ⚙️ Technologies Used
- Python 3
- TensorFlow / Keras
- NumPy, pandas, matplotlib, seaborn
- OpenCV
- Multi-Head Attention (Transformer Layer)
- Scikit-learn

## 🤝 Contributing
Contributions are welcome! Here's how you can help:

💡 Suggest new features or improvements.

🐛 Report bugs or issues in the model or dataset handling.

📊 Add evaluation metrics, visualizations, or experiment with different architectures.
