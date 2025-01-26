# Phone Price Prediction Using Neural Networks

## Project Overview
This project aims to predict the price range of mobile phones based on various features such as battery power, RAM, internal memory, etc. A neural network model is implemented using TensorFlow and Keras to classify phones into different price categories.

## Dataset
The dataset consists of several numerical features related to phone specifications. Some key features include:
- `battery_power`: Total energy a battery can store in one charge.
- `ram`: Random Access Memory in MB.
- `int_memory`: Internal memory in GB.
- `mobile_wt`: Weight of the phone in grams.
- `px_height` and `px_width`: Pixel resolution of the phone.

### Target Variable
- `price_range`: A categorical variable indicating the price category (0, 1, 2, 3).

## Prerequisites
Make sure you have the following dependencies installed:

```bash
pip install numpy pandas scikit-learn tensorflow keras imbalanced-learn
```

## Project Structure
```
.
├── data
│   ├── mobile_price_train.csv
│   ├── mobile_price_test.csv
├── model
│   ├── phone_price_prediction_model.h5
├── notebooks
│   ├── exploratory_data_analysis.ipynb
│   ├── model_training.ipynb
├── scripts
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate_model.py
├── README.md
```

## Data Preprocessing
Before training the model, the data is preprocessed by:
1. Handling missing values (if any).
2. Scaling numerical features using `StandardScaler`.
3. Splitting data into training and testing sets.
4. Addressing class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

## Model Architecture
The model consists of the following layers:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, input_dim=number_of_features, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### Compilation
```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### Training
```python
model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test))
```

## Model Evaluation
After training, the model is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score

## Running the Project
1. Preprocess the data:
   ```bash
   python scripts/preprocess.py
   ```
2. Train the model:
   ```bash
   python scripts/train_model.py
   ```
3. Evaluate the model:
   ```bash
   python scripts/evaluate_model.py
   ```

## Results
Once trained, the model should ideally achieve an accuracy of over 85% on the test data. If the model underperforms, consider tuning hyperparameters such as:
- Number of layers and neurons.
- Learning rate.
- Number of epochs.
- Batch size.

## Future Improvements
- Try different architectures like Convolutional Neural Networks (CNNs).
- Experiment with additional features or feature engineering techniques.
- Use GridSearchCV or RandomizedSearchCV for hyperparameter tuning.

## Conclusion
This project successfully predicts phone price categories using a neural network model, with potential applications in e-commerce and market analysis.

---

## Author
Omash Viduranga

