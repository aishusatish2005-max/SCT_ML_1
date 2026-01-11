import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
train = pd.read_csv("train.csv")   # change filename if needed

# Preview data
print("Training data preview:")
print(train.head())

# Clean column names (IMPORTANT)
train.columns = train.columns.str.strip().str.replace('.', '', regex=False)

print("\nColumns in dataset:")
print(train.columns)

# Select features and target
features = [
    'SQUARE_FT',
    'BHK_NO',
    'UNDER_CONSTRUCTION',
    'RERA',
    'READY_TO_MOVE',
    'RESALE',
    'LONGITUDE',
    'LATITUDE'
]

X = train[features]
y = train['TARGET(PRICE_IN_LACS)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Sample prediction
sample = X.iloc[[0]]
predicted_price = model.predict(sample)
print("\nPredicted price for first house:", predicted_price[0], "Lakhs")
