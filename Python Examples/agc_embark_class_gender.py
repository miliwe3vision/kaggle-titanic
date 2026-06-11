# Titanic Survival Prediction using Logistic Regression
# Python 3 Version

import pandas as pd
import statsmodels.formula.api as smf

# ==========================================
# Load Data
# ==========================================

print("Loading data...")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ==========================================
# Data Cleaning
# ==========================================

# Fill missing Age values
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

# Fill missing Embarked values
train["Embarked"] = train["Embarked"].fillna(
    train["Embarked"].mode()[0]
)

test["Embarked"] = test["Embarked"].fillna(
    train["Embarked"].mode()[0]
)

# Fill missing Fare values
test["Fare"] = test["Fare"].fillna(
    test["Fare"].median()
)

# ==========================================
# Build Logistic Regression Model
# ==========================================

print("Training model...")

formula = """
Survived ~ C(Pclass)
          + C(Sex)
          + Age
          + SibSp
          + C(Embarked)
"""

model = smf.logit(
    formula=formula,
    data=train
)

results = model.fit()

print("\nModel Summary:\n")
print(results.summary())

# ==========================================
# Predict Survival Probabilities
# ==========================================

print("\nMaking predictions...")

predicted_probabilities = results.predict(test)

# Convert probabilities to classes
predicted_survival = (
    predicted_probabilities >= 0.5
).astype(int)

# ==========================================
# Create Submission File
# ==========================================

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predicted_survival
})

submission.to_csv(
    "results_embarkclassgendermodel.csv",
    index=False
)

print("\nSubmission file created:")
print("results_embarkclassgendermodel.csv")

# ==========================================
# Display First Few Predictions
# ==========================================

print("\nSample Predictions:")
print(submission.head())

print("\nAnalysis Ended")