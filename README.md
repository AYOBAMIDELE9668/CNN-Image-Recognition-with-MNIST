# CNN-Image-Recognition-with-MNIST
A deep learning project that uses a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The project walks through Exploratory Data Analysis (EDA), preprocessing, model building, training, and evaluation using TensorFlow/Keras.
Project Objectives

- Understand the MNIST dataset through visualization and statistical summaries.
- Preprocess image data to make it model-ready.
- Build a Convolutional Neural Network from scratch using Keras.
- Train, evaluate, and visualize the performance of the model.
- Test with real-world handwritten digit data to simulate deployment scenarios.

---

##  Exploratory Data Analysis (EDA)

The notebook starts with a comprehensive EDA:

- **Dataset Description**: The MNIST dataset contains 70,000 grayscale images (28x28 pixels) of handwritten digits (0–9).
- **Shape & Structure**: Each image is a 28x28 pixel grid, flattened to 784 features per sample.
- **Label Distribution**: A balanced dataset with approximately 7,000 samples per digit class.
- **Sample Visualization**: Random samples of digits are displayed with their labels for visual confirmation.
- **Pixel Intensity Distribution**: Heatmaps and histograms explore the variation in pixel intensity across digits.

---

##  Data Preprocessing

- **Normalization**: Pixel values are scaled to range [0, 1] by dividing by 255.
- **Reshaping**: Images are reshaped to `(28, 28, 1)` to match the CNN input shape.
- **One-hot Encoding**: Target labels are converted to categorical format.
- **Train/Test Split**: Dataset is divided into training and testing sets (typically 80/20 split).

## Model Architecture

The CNN was built using the Keras Sequential API with the following layers:

```python
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
##  Model Compilation & Training

- **Loss Function**: `categorical_crossentropy`  
- **Optimizer**: `Adam`  
- **Metrics**: `accuracy`  
- **Batch Size**: `128`  
- **Epochs**: `10–20` (customizable based on performance)

Training and validation performance were monitored using **accuracy and loss curves** plotted after each epoch to observe learning trends.

## Model Evaluation

- **Test Accuracy**: Achieved approximately **98% accuracy** on the MNIST test set.

### Metrics Evaluated:

- **Confusion Matrix**  
  Visual representation of the predicted vs actual labels to identify which digits were commonly misclassified.

- **Classification Report**  
  Provides a breakdown of:
  - **Precision**
  - **Recall**
  - **F1-score**
  - **Support**  
  for each class (digit 0–9).
## Real-world Testing

To test the model in a practical scenario, real-world handwritten digits were introduced:

- **Image Preprocessing**:
  - Resized images to **28x28**
  - Converted to **grayscale**
  - Normalized and reshaped to fit model input

- **Model Inference**:
  - Predictions were made using the trained model.
  - Output included **predicted digit** and **confidence probabilities**.

- **Observation**:
  - The model successfully classified most real-world digits with high confidence.
  - Minor drop in accuracy was observed when handwriting styles were significantly different from MNIST samples.

##  Key Takeaways

-  **CNNs** are highly effective in learning and extracting features from image data, making them ideal for digit recognition.
-  Even a **simple CNN architecture** (with dropout layers) can achieve very high performance on MNIST.
-  With proper preprocessing, the model **generalizes well to real-world handwritten digits**.
-  **Exploratory Data Analysis (EDA)** is critical before model training — it reveals important data patterns, class imbalances, and prepares the ground for effective modeling.
