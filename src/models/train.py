# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from xgboost import XGBClassifier
# import joblib
# import os

# def train_model(X, y):
#     # Create model directory if it doesn't exist
#     os.makedirs("model", exist_ok=True)

#     # Split the data for validation
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     # Define models and their parameter grids
#     models = {
#         "RandomForest": {
#             "model": RandomForestClassifier(),
#             "params": {
#                 "n_estimators": [100, 200],
#                 "max_depth": [None, 10, 20]
#             }
#         },
#         "LogisticRegression": {
#             "model": LogisticRegression(max_iter=1000),
#             "params": {
#                 "C": [0.1, 1, 10]
#             }
#         },
#         "SVC": {
#             "model": SVC(),
#             "params": {
#                 "C": [0.5, 1, 10],
#                 "kernel": ['linear', 'rbf']
#             }
#         },
#         "GradientBoosting": {
#             "model": GradientBoostingClassifier(),
#             "params": {
#                 "n_estimators": [100, 200],
#                 "learning_rate": [0.05, 0.1]
#             }
#         },
#         "XGBoost": {
#             "model": XGBClassifier(eval_metric='logloss'),
#             "params": {
#                 "n_estimators": [100, 200],
#                 "max_depth": [3, 6],
#                 "learning_rate": [0.05, 0.1]
#             }
#         }
#     }

#     best_score = 0
#     best_model = None
#     best_name = ""

#     # Loop through each model and perform GridSearchCV
#     for name, m in models.items():
#         print(f"Training and tuning {name}...")
#         grid = GridSearchCV(m["model"], m["params"], cv=5, scoring='accuracy', n_jobs=-1)
#         grid.fit(X_train, y_train)

#         y_pred = grid.predict(X_val)
#         acc = accuracy_score(y_val, y_pred)

#         print(f"{name} Accuracy: {acc:.4f}")
#         print(classification_report(y_val, y_pred))

#         if acc > best_score:
#             best_score = acc
#             best_model = grid.best_estimator_
#             best_name = name

#     # Save the best model
#     joblib.dump(best_model, "model/model.pkl")
#     print(f"✅ Best Model: {best_name} with accuracy: {best_score:.4f} (Saved to model/model.pkl)")

#     return best_model


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
import os

def train_model(X, y):
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)

    # Split the data with stratification
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Calculate scale_pos_weight for XGBoost
    counter = Counter(y_train)
    scale_pos_weight = counter[0] / counter[1]  # adjust if your classes are 0 (majority), 1 (minority)

    # Define models with balanced handling
    models = {
        "RandomForest": {
            "model": RandomForestClassifier(class_weight='balanced', random_state=42),
            "params": {
                "n_estimators": [100, 150],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
    }
        },
        "LogisticRegression": {
            "model": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            "params": {
                "C": [0.1, 1, 10]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.1]
            }
        }
    }

    best_score = 0
    best_model = None
    best_name = ""

    # Train and tune each model
    for name, m in models.items():
        print(f"\n🔍 Training and tuning {name}...")
        grid = GridSearchCV(m["model"], m["params"], cv=5, scoring='f1_weighted', n_jobs=-1)
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        train_acc = accuracy_score(y_train, grid.predict(X_train))
        f1 = f1_score(y_val, y_pred, average='weighted')

        print(f"\n📊 {name} Results:")
        print(f"Accuracy      : {acc:.4f}")
        print(f"Train Accuracy      : {train_acc:.4f}")
        print(f"F1 Score      : {f1:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
        print("Classification Report:\n", classification_report(y_val, y_pred))

        if f1 > best_score:
            best_score = f1
            best_model = grid.best_estimator_
            best_name = name

    # Save the best model
    joblib.dump(best_model, "model/model.pkl")
    print(f"\n✅ Best Model: {best_name} with F1 score: {best_score:.4f} (Saved to model/model.pkl)")

    return best_model
