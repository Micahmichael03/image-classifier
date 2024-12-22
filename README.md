# image-classifier

---
![Classifier Pipeline](image-classifier/Screenshot 2024-12-20 144108.png)

# Binary Image Classifier Project

## Overview

This project implements a binary image classifier to distinguish between two categories: "empty" and "not_empty." The complete pipeline involves data preprocessing, model training, hyperparameter tuning, and model evaluation using an SVM (Support Vector Machine) classifier.

## Files Explained

### Main Script

The main script performs the following steps:

1. **Importing Necessary Libraries**: 
   - `os` for handling file paths.
   - `numpy` for array operations.
   - Image processing tools from `skimage`.
   - Machine learning tools from `scikit-learn` for data splitting, hyperparameter tuning, and training an SVM model.

2. **Defining Dataset Directory and Categories**: 
   - Specifies the directory containing the dataset.
   - Defines categories for classification ("empty" and "not_empty").

3. **Initializing Data Structures**:
   - Initializes `data` and `labels` lists to store processed image data and corresponding labels.

4. **Processing Images**:
   - Iterates through each category folder.
   - Constructs file paths for each image.
   - Resizes each image to a fixed dimension of 15x15 pixels.
   - Flattens the image into a one-dimensional array and appends it to the `data` list.
   - Appends corresponding labels to the `labels` list.
   - Handles errors during processing gracefully.

5. **Converting and Splitting Data**:
   - Converts `data` and `labels` lists into NumPy arrays.
   - Splits data into training and testing sets (80% training, 20% testing) using stratified sampling to ensure balanced class representation.

6. **Model Training and Hyperparameter Tuning**:
   - Utilizes an SVM model.
   - Performs hyperparameter tuning using grid search to find the best combination of `gamma` and `C`.
   - Trains the model on the training data.
   - Evaluates model performance on the test set using accuracy as the metric.

7. **Saving the Model**:
   - Saves the best model from the grid search to a file named `model.pkl` using `pickle` for future use without retraining.

### Dataset Folder Structure

The dataset folder is structured as follows:

- **Dataset Root Folder (`clf-data`)**: This is the main directory containing the data for training and testing.
  - **Subfolder: `empty`**: Contains images representing the "empty" class. These images likely depict scenes or objects without certain features.
  - **Subfolder: `not_empty`**: Contains images representing the "not_empty" class. These images differ from the "empty" category based on the presence of specific features.

Each subfolder contains image files in a compatible format, such as JPEG or PNG. The images are varied to accurately represent the respective categories for effective training and testing of the classifier.

## Usage Instructions

1. **Install Dependencies**:
   ```bash
   pip install numpy scikit-image scikit-learn
   ```

2. **Run the Main Script**:
   ```bash
   python main_script.py
   ```

3. **Model Evaluation**:
   - After running the script, the best model will be saved as `model.pkl`.
   - Use this model for future predictions on new image data without retraining.

## Contributing

Feel free to fork this project, make improvements, and submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any questions or inquiries, please contact the project maintainer at [makoflash05@gmai.com].

---
