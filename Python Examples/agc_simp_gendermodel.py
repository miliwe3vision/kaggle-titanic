import csv
import numpy as np

data = []

# ==========================
# Read Training Data
# ==========================
with open('train.csv', 'r', newline='', encoding='utf-8') as f:
    csv_file = csv.reader(f)

    # Skip header
    next(csv_file)

    for row in csv_file:
        data.append(row)

# Convert to NumPy array
data = np.array(data)

# ==========================
# Separate by Gender
# ==========================
women_only_stats = data[:, 3] == "female"
men_only_stats = data[:, 3] == "male"

# ==========================
# Survival Statistics
# ==========================
women_onboard = data[women_only_stats, 0].astype(float)
men_onboard = data[men_only_stats, 0].astype(float)

proportion_women_survived = (
    np.sum(women_onboard) / len(women_onboard)
)

proportion_men_survived = (
    np.sum(men_onboard) / len(men_onboard)
)

proportion_survivors = (
    np.sum(data[:, 0].astype(float))
    / len(data[:, 0])
)

# ==========================
# Print Results
# ==========================
print(f"Proportion of people who survived is {proportion_survivors:.4f}")
print(f"Proportion of women who survived is {proportion_women_survived:.4f}")
print(f"Proportion of men who survived is {proportion_men_survived:.4f}")

# ==========================
# Gender-Based Predictions
# Female -> Survived (1)
# Male -> Died (0)
# ==========================
with open('train.csv', 'r', newline='', encoding='utf-8') as f_in, \
     open('train_results_genderbasedmodelpy.csv', 'w', newline='', encoding='utf-8') as f_out:

    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    # Skip header
    next(reader)

    for row in reader:

        if row[3] == 'female':
            row[0] = '1'
        else:
            row[0] = '0'

        writer.writerow(row)

print("Analysis ended")