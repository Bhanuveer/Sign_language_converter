import numpy as np
import pickle
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
with open('./data.pickle', 'rb') as file:
    data_dict = pickle.load(file)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check label distribution
label_counts = Counter(labels)
print("Label distribution before filtering:", label_counts)

# Filter out classes with fewer than 2 instances
filtered_data = []
filtered_labels = []
for label, count in label_counts.items():
    if count >= 2:
        filtered_data.extend(data[labels == label])
        filtered_labels.extend(labels[labels == label])

filtered_data = np.array(filtered_data)
filtered_labels = np.array(filtered_labels)

# Check label distribution after filtering
label_counts = Counter(filtered_labels)
print("Label distribution after filtering:", label_counts)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    filtered_data, filtered_labels, test_size=0.2, shuffle=True, stratify=filtered_labels)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict
y_predict = model.predict(x_test)

# Evaluate
score = accuracy_score(y_predict, y_test)
print(f'{score * 100}% of samples were classified correctly!')

print(classification_report(y_test, y_predict))

"""# Save model and label mapping
label_mapping = {i: chr(65 + i) for i in range(26)}
label_mapping[26] = 'OK'  # Add OK gesture"""
label_mapping = {0: "OK", 1: "HELLO", 2:'I Love You',3: 'Telephone',4: 'Money',5: 'Love',6: 'yes',7: 'No',8: 'bathroom',9: 'Iam',10: 'why',11: 'water',12: 'you',13: 'errase'}

with open('model2.p', 'wb') as file:
    pickle.dump({'model': model, 'label_mapping': label_mapping}, file)