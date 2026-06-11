import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train_data = []
test_data = []

# ==========================
# Load Training Data
# ==========================
with open('train.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        train_data.append(row)

train_data = np.array(train_data)

# ==========================
# Load Test Data
# ==========================
with open('test.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        test_data.append(row)

test_data = np.array(test_data)

# ==========================
# TRAINING DATA CLEANING
# ==========================

# Sex
train_data[train_data[:, 3] == 'male', 3] = '1'
train_data[train_data[:, 3] == 'female', 3] = '0'

# Embarked
train_data[train_data[:, 10] == 'C', 10] = '0'
train_data[train_data[:, 10] == 'S', 10] = '1'
train_data[train_data[:, 10] == 'Q', 10] = '2'

# Missing Age
median_age = np.median(
    train_data[train_data[:, 4] != '', 4].astype(float)
)

train_data[train_data[:, 4] == '', 4] = str(median_age)

# Missing Embarked
mean_embarked = round(
    np.mean(
        train_data[train_data[:, 10] != '', 10].astype(float)
    )
)

train_data[train_data[:, 10] == '', 10] = str(mean_embarked)

# Remove Name, Ticket, Cabin
train_data = np.delete(train_data, [2, 7, 9], axis=1)

# ==========================
# TEST DATA CLEANING
# ==========================

# Sex
test_data[test_data[:, 2] == 'male', 2] = '1'
test_data[test_data[:, 2] == 'female', 2] = '0'

# Embarked
test_data[test_data[:, 9] == 'C', 9] = '0'
test_data[test_data[:, 9] == 'S', 9] = '1'
test_data[test_data[:, 9] == 'Q', 9] = '2'

# Missing Age
median_age_test = np.median(
    test_data[test_data[:, 3] != '', 3].astype(float)
)

test_data[test_data[:, 3] == '', 3] = str(median_age_test)

# Missing Embarked
median_embarked = round(
    np.median(
        test_data[test_data[:, 9] != '', 9].astype(float)
    )
)

test_data[test_data[:, 9] == '', 9] = str(median_embarked)

# Missing Fare
for i in range(len(test_data)):
    if test_data[i, 7] == '':

        same_class = (
            (test_data[:, 7] != '') &
            (test_data[:, 0] == test_data[i, 0])
        )

        median_fare = np.median(
            test_data[same_class, 7].astype(float)
        )

        test_data[i, 7] = str(median_fare)

# Remove Name, Ticket, Cabin
test_data = np.delete(test_data, [1, 6, 8], axis=1)

# ==========================
# Convert to Numeric
# ==========================

X_train = train_data[:, 1:].astype(float)
y_train = train_data[:, 0].astype(int)

X_test = test_data.astype(float)

# ==========================
# Train Model
# ==========================

print("Training...")

forest = RandomForestClassifier(
    n_estimators=1000,
    random_state=42
)

forest.fit(X_train, y_train)

# ==========================
# Predict
# ==========================

print("Predicting...")

output = forest.predict(X_test)

# ==========================
# Save Submission
# ==========================

with open("agcfirstforest.csv", "w", newline="") as submission:
    writer = csv.writer(submission)

    writer.writerow(["Survived"] + header[1:])

    with open("test.csv", "r", newline='', encoding='utf-8') as test_file:
        reader = csv.reader(test_file)
        next(reader)

        for prediction, row in zip(output, reader):
            writer.writerow([int(prediction)] + row)

print("Analysis has Finished")