#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#getting data
raw_data=pd.read_csv('datasetfoodallergy123.csv')
raw_data.head()


# In[73]:


# Function to update the dataset
def update_dataset():
    # Prompt the user for input
    food_product = input("Enter the food product: ")
    user_allergic_reactions = input("Enter allergic reactions (comma-separated): ")

    # Map user input to food_id
    food_id = raw_data.loc[raw_data['food_product'] == food_product]['food_id'].values[0]

    # Rule-based matching
    allergic_reactions = user_allergic_reactions.split(', ')
    allergic_columns = ['allergic_reaction_stomachache', 'allergic_reaction_nausea', 'allergic_reaction_diarrhea', 'allergic_reaction_swelling', 'allergic_reaction_hives', 'allergic_reaction_itching']

    for symptom in allergic_reactions:
        if symptom in allergic_columns:
            raw_data.loc[raw_data['food_id'] == food_id, symptom] = 1

    # Calculate the likelihood based on symptoms and allergens
    likelihood = raw_data.loc[raw_data['food_id'] == food_id, allergic_columns + ['allergen_peanut', 'allergen_soy', 'allergen_gluten', 'allergen_dairy', 'allergen_fish']].product().values[0]

    # Update the prediction column based on likelihood
    raw_data.loc[raw_data['food_id'] == food_id, 'prediction'] = 1 if likelihood > 0 else 0

    # Print the nutritional data
    nutritional_data = raw_data.loc[raw_data['food_id'] == food_id, 'nutritional_data'].values[0]
    print(f"Likelihood of allergic reaction: {likelihood}")
    print(f"Nutritional data: {nutritional_data}")

# Run the function to update the dataset based on user input
update_dataset()

raw_data.describe()


# In[74]:


#class of user allergic reactions
raw_data['user_allergic_reactions'].value_counts().plot(kind = 'bar', title = 'Allergy Distribution')


# In[75]:


#food products
plt.figure(figsize=(12, 6))
raw_data['food_product'].value_counts().plot(kind = 'bar', title = 'Food_Products')
plt.show()


# In[76]:


#missing values
missing_values = raw_data.isnull().sum()
print(missing_values)
print("\n")
#handling missing values
raw_data['prediction'].fillna(0, inplace=True)  # Assuming 0 means no allergic reaction
print("Data Summary After Handling Missing Values:")
print(raw_data.describe())


# In[85]:


# Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Split dataset
feature_cols = ['allergic_reaction_stomachache', 'allergic_reaction_nausea', 'allergic_reaction_diarrhea', 'allergic_reaction_swelling', 'allergic_reaction_hives', 'allergic_reaction_itching']
x = raw_data[feature_cols]
y = raw_data['prediction']

# Split x and y into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=16)

# Instantiate the model
logreg = LogisticRegression(random_state=16)

# Fit the model with data
logreg.fit(x_train, y_train)

# Predict
y_pred = logreg.predict(x_test)

# Measure accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nLogistic Regression Accuracy:", accuracy)

# Confusion matrix
confusion = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Logistic Regression')

# Calculate percentages for actual and predicted
total_entries = len(y_test)
actual_percentages = (y_test.value_counts() / total_entries) * 100
predicted_percentages = (pd.Series(y_pred).value_counts() / total_entries) * 100

# Create the bar graph
plt.figure(figsize=(12, 9))
bar_width = 0.30
index = range(len(actual_percentages))

# Plot actual and predicted percentages
plt.bar(index, actual_percentages, bar_width, label='Actual', color='skyblue', alpha=0.7)
plt.bar([i + bar_width for i in index], predicted_percentages, bar_width, label='Predicted', color='orange', alpha=0.7)

# Add labels
plt.xlabel('Categories', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.title('Comparison of Actual and Predicted Target Variable Percentages for Logistic Regression', fontsize=16)
plt.xticks([i + bar_width / 2 for i in index], actual_percentages.index, rotation=0)
plt.legend()

# Add percentage labels on top of the bars
for bars, percentages in zip([plt.bar(index, actual_percentages, bar_width, label='Actual', color='skyblue', alpha=0.7),
                              plt.bar([i + bar_width for i in index], predicted_percentages, bar_width, label='Predicted', color='orange', alpha=0.7)],
                             [actual_percentages, predicted_percentages]):
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        plt.annotate(f'{percentage:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=12)

plt.tight_layout()

# Show the plot
plt.show()

# Performance Metrics
# Calculate the confusion matrix for Logistic Regression
confusion_lr = confusion_matrix(y_test, y_pred)

# Calculate the precision, recall, and F1 score
true_positives = confusion_lr[1, 1]
false_positives = confusion_lr[0, 1]
false_negatives = confusion_lr[1, 0]

precision_lr = true_positives / (true_positives + false_positives)
recall_lr = true_positives / (true_positives + false_negatives)
f1_lr = 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr)

print("Logistic Regression Precision:", precision_lr)
print("Logistic Regression Recall:", recall_lr)
print("Logistic Regression F1 Score:", f1_lr)


# In[86]:


#Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Instantiate the Multinomial Naive Bayes model
nb = MultinomialNB()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=16)

# Fit the model with data
nb.fit(x_train, y_train)

# Make predictions on the test set
y_pred = nb.predict(x_test)

# Calculate the confusion matrix for Naive Bayes
confusion_nb = confusion_matrix(y_test, y_pred)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_nb, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Naive Bayes')

# Show the plot
plt.show()

# Measure the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", accuracy)

# Create a list of indices for the test data
indices = range(len(y_test))

# Create subplots
plt.figure(figsize=(10, 5))

# Plot the actual values
plt.bar(indices, y_test, width=0.4, align='center', alpha=0.7, label='Actual', color='b')

# Plot the predicted values
plt.bar(indices, y_pred, width=0.4, align='edge', alpha=0.7, label='Predicted', color='g')

# Add labels and a legend
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.title('Actual vs. Predicted Values for Naive Bayes')
plt.legend()

# Show the plot
plt.show()

# Calculate the precision, recall, and F1 score for Naive Bayes
true_positives = confusion_nb[1, 1]
false_positives = confusion_nb[0, 1]
false_negatives = confusion_nb[1, 0]

precision_nb = true_positives / (true_positives + false_positives)
recall_nb = true_positives / (true_positives + false_negatives)
f1_nb = 2 * (precision_nb * recall_nb) / (precision_nb + recall_nb)

print("Naive Bayes Precision:", precision_nb)
print("Naive Bayes Recall:", recall_nb)
print("Naive Bayes F1 Score:", f1_nb)


# In[87]:


#SVM
from sklearn.svm import SVC

# Instantiate the Support Vector Machine model
svm = SVC(random_state=16)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)

# Fit the SVM model with data
svm.fit(x_train, y_train)

# Make predictions on the test set
y_pred_svm = svm.predict(x_test)

# Calculate the confusion matrix for SVM
confusion_svm = confusion_matrix(y_test, y_pred_svm)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_svm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Support Vector Machine (SVM)')

# Show the plot
plt.show()

# Create a list of indices for the test data
indices = range(len(y_test))

# Create subplots
plt.figure(figsize=(10, 5))

# Plot the actual values
plt.bar(indices, y_test, width=0.4, align='center', alpha=0.7, label='Actual', color='b')

# Plot the predicted values
plt.bar(indices, y_pred_svm, width=0.4, align='edge', alpha=0.7, label='Predicted', color='g')

# Add labels and a legend
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.title('Actual vs. Predicted Values for Support Vector Machine (SVM)')
plt.legend()

# Show the plot
plt.show()

# Measure the accuracy of the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Calculate the precision, recall, and F1 score for SVM
true_positives_svm = confusion_svm[1, 1]
false_positives_svm = confusion_svm[0, 1]
false_negatives_svm = confusion_svm[1, 0]

precision_svm = true_positives_svm / (true_positives_svm + false_positives_svm)
recall_svm = true_positives_svm / (true_positives_svm + false_negatives_svm)
f1_svm = 2 * (precision_svm * recall_svm) / (precision_svm + recall_svm)

print("SVM Precision:", precision_svm)
print("SVM Recall:", recall_svm)
print("SVM F1 Score:", f1_svm)


# In[96]:


import matplotlib.pyplot as plt
import numpy as np

# Performance Metrics for Each Model
models = ['Logistic Regression', 'Naive Bayes', 'SVM']
precision = [0.625, 0.833, 0.714]
recall = [0.4545, 0.4545, 0.8182]
f1_score = [0.5263, 0.5882, 0.7632]

# Define the width of each bar
bar_width = 0.2

# Create an array of x-coordinates for each group of bars
x = np.arange(len(models))

# Create subplots
plt.figure(figsize=(12, 6))

# Plot precision values
plt.bar(x - bar_width, precision, width=bar_width, align='center', label='Precision', alpha=0.7, color='blue')

# Plot recall values
plt.bar(x, recall, width=bar_width, align='center', label='Recall', alpha=0.7, color='green')

# Plot F1 score values
plt.bar(x + bar_width, f1_score, width=bar_width, align='center', label='F1 Score', alpha=0.7, color='orange')

# Set the x-axis labels to be the model names
plt.xticks(x, models)

# Add labels and a legend
plt.xlabel('Models')
plt.ylabel('Performance Metrics')
plt.title('Performance Metrics for Different Models')
plt.legend(loc='best')

# Add value labels on top of the bars
for i, value in enumerate(precision):
    plt.text(i - bar_width, value + 0.02, f'{value:.4f}', ha='center', va='bottom', fontsize=10)
for i, value in enumerate(recall):
    plt.text(i, value + 0.02, f'{value:.4f}', ha='center', va='bottom', fontsize=10)
for i, value in enumerate(f1_score):
    plt.text(i + bar_width, value + 0.02, f'{value:.4f}', ha='center', va='bottom', fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()


# In[1]:


#logistic
import matplotlib.pyplot as plt

# Performance Metrics for Logistic Regression
metrics = ['Precision', 'Recall', 'F1 Score']
values = [0.625, 0.4545, 0.5263]

# Create an array of x-coordinates for each group of bars
x = range(len(metrics))

# Create subplots
plt.figure(figsize=(8, 6))

# Plot the performance metrics
plt.bar(x, values, width=0.4, align='center', alpha=0.7, color='blue')

# Set the x-axis labels to be the metric names
plt.xticks(x, metrics)

# Add labels and a legend
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Metrics for Logistic Regression')

# Add value labels on top of the bars
for i, value in enumerate(values):
    plt.text(i, value + 0.02, f'{value:.4f}', ha='center', va='bottom', fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()


# In[2]:


#naive bayes
import matplotlib.pyplot as plt

# Performance Metrics for Naive Bayes
metrics = ['Precision', 'Recall', 'F1 Score']
values = [0.833, 0.4545, 0.5882]

# Create an array of x-coordinates for each group of bars
x = range(len(metrics))

# Create subplots
plt.figure(figsize=(8, 6))

# Plot the performance metrics
plt.bar(x, values, width=0.4, align='center', alpha=0.7, color='green')

# Set the x-axis labels to be the metric names
plt.xticks(x, metrics)

# Add labels and a legend
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Metrics for Naive Bayes')

# Add value labels on top of the bars
for i, value in enumerate(values):
    plt.text(i, value + 0.02, f'{value:.4f}', ha='center', va='bottom', fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()


# In[3]:


#SVM
import matplotlib.pyplot as plt

# Performance Metrics for SVM
metrics = ['Precision', 'Recall', 'F1 Score']
values = [0.714, 0.8182, 0.7632]

# Create an array of x-coordinates for each group of bars
x = range(len(metrics))

# Create subplots
plt.figure(figsize=(8, 6))

# Plot the performance metrics
plt.bar(x, values, width=0.4, align='center', alpha=0.7, color='orange')

# Set the x-axis labels to be the metric names
plt.xticks(x, metrics)

# Add labels and a legend
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Metrics for SVM')

# Add value labels on top of the bars
for i, value in enumerate(values):
    plt.text(i, value + 0.02, f'{value:.4f}', ha='center', va='bottom', fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:




