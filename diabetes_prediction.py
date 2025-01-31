import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

pd.set_option('future.no_silent_downcasting', True)
diabetes_data = pd.read_csv("diabetes.csv")

X = diabetes_data.iloc[:, :-1]
Y = diabetes_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


classifier_model = SVC(kernel="linear")
classifier_model.fit(X_train, y_train)

# Evaluate the model
X_train_pred = classifier_model.predict(X_train)
X_test_pred = classifier_model.predict(X_test)

# Accuracy score
train_data_accuracy = accuracy_score(X_train_pred, y_train)
test_data_accuracy = accuracy_score(X_test_pred, y_test)

# Compute how well we performed
correct = (y_test == X_test_pred).sum()
incorrect = (y_test != X_test_pred).sum()


# Print results
# print(f"Results for model {type(classifier_model).__name__}")
# print(f"Correct: {correct}")
# print(f"Incorrect: {incorrect}")
# print(f"Accuracy: {test_data_accuracy * 100:.2f}%")


# Example: 
# Define the input data
# input_data = (6, 148, 72, 35, 0, 33.6, 0.627, 50)
# input_data = (1, 85, 66, 29, 0, 26.6, 0.351, 31)

# # Convert input data to a DataFrame with correct column names
# column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
#                 'BMI', 'DiabetesPedigreeFunction', 'Age']

# input_data_df = pd.DataFrame([input_data], columns=column_names)

# # Standardize the input data
# std_data = scaler.transform(input_data_df)

# # Convert back to DataFrame to retain feature names
# # std_data_df = pd.DataFrame(std_data, columns=column_names)

# # Make the prediction
# prediction = classifier_model.predict(std_data)


# # Output the result
# if prediction[0] == 0:
#     print("The person is not diabetic.")
# else:
#     print("The person is diabetic.")



