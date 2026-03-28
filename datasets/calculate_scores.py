import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# ---------- COMMON FUNCTION ----------
def evaluate_logistic_regression(df, target_col, dataset_name):
    print(f"\n===== {dataset_name.upper()} DATASET =====")

    # Drop rows with missing target or features
    df = df.dropna(subset=[target_col])
    df = df.dropna(axis=1, how='all')
    df = df.dropna()

    # Encode categorical columns safely
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df.loc[:, col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Ensure target is binary or numeric
    if y.dtype == 'object':
        y = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Handle binary/multiclass cases
    avg_type = 'binary' if len(set(y)) == 2 else 'weighted'

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=avg_type, zero_division=0)
    rec = recall_score(y_test, y_pred, average=avg_type, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg_type, zero_division=0)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")

# ---------- FRAMINGHAM ----------
framingham = pd.read_csv("framingham.csv")
evaluate_logistic_regression(framingham, "TenYearCHD", "Framingham")

# ---------- HEART DISEASE ----------
heart = pd.read_csv("heart_disease_dataset.csv")
target_col = "target" if "target" in heart.columns else heart.columns[-1]
evaluate_logistic_regression(heart, target_col, "Heart Disease")

# ---------- BREAST CANCER ----------
cancer_data = load_breast_cancer()
cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
cancer["target"] = cancer_data.target
evaluate_logistic_regression(cancer, "target", "Breast Cancer")

# ---------- MENTAL HEALTH (Simulated Example) ----------
# If you have an actual file, replace this section with:
mental = pd.read_csv("mental_health_survey.csv")
# mental = pd.DataFrame({
#     "age": [22, 30, 25, 40, 50, 28, 33, 45, 29, 55],
#     "stress": [7, 5, 6, 8, 9, 4, 5, 7, 6, 8],
#     "sleep_hours": [5, 7, 6, 4, 3, 8, 7, 5, 6, 4],
#     "exercise": [0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
#     "mental_illness": [1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
# })
evaluate_logistic_regression(mental, "mental_illness", "Mental Health")
