import os
import numpy as np
import pickle
from skimage.io import imread  # Used for reading images
from skimage.transform import resize  # Used for resizing images
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.metrics import accuracy_score  # For evaluating model performance

# Preparing the dataset
input_dir = r'C:\Users\user\OneDrive\Documents\Computer Vision\Beginner_projects\image-classifier\clf-data'
categories = ['empty', 'not_empty']  # Categories to classify

data = []  # List to store image data
labels = []  # List to store corresponding labels

# Loop through each category and process its images
for category in categories:
    category_path = os.path.join(input_dir, category)  # Get the path for the current category
    for file in os.listdir(category_path):  # Iterate through all files in the category
        img_path = os.path.join(category_path, file)  # Construct the full path for the image file
        print(f"Processing file: {img_path}")  # Debugging message to confirm file path
        if not os.path.isfile(img_path):  # Check if the path points to a valid file
            continue  # Skip directories or invalid paths
        try:
            img = imread(img_path)  # Read the image from the file
            img = resize(img, (15, 15))  # Resize the image to a fixed 15x15 dimension
            data.append(img.flatten())  # Flatten the 2D image into a 1D array and add to data
            labels.append(categories.index(category))  # Use the category index as the label
        except Exception as e:
            print(f"Error processing file {img_path}: {e}")  # Print an error message if processing fails

# Convert the data and labels lists into NumPy arrays
data = np.asarray(data)
labels = np.asarray(labels)

# Debugging: Print the shapes of the dataset and labels to confirm successful loading
print(f"Data shape: {data.shape}")  # Should show (num_samples, num_features)
print(f"Labels shape: {labels.shape}")  # Should show (num_samples,)

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)  # Use 80% for training and 20% for testing

# Train the SVM classifier
classifier = SVC()  # Initialize a basic SVM classifier

# Define a range of hyperparameters for grid search
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(classifier, parameters)  # Initialize grid search
grid_search.fit(x_train, y_train)  # Train the model with grid search on the training data

# Test the performance of the best model
best_estimator = grid_search.best_estimator_  # Get the model with the best hyperparameters
y_prediction = best_estimator.predict(x_test)  # Predict labels for the testing data
score = accuracy_score(y_test, y_prediction)  # Calculate accuracy of the predictions

# Print the accuracy of the model
print(f"Accuracy: {score}")

# Save the trained model to a file
with open('model.pkl', 'wb') as f:  # Open a file in write-binary mode
    pickle.dump(best_estimator, f)  # Save the best model using pickle
    print("Model saved to model.pkl")  # Confirm that the model was saved successfully
