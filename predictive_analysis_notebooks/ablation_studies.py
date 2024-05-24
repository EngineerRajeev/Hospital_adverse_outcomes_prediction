import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_data(df, drop_features=[]):
    if drop_features:
        df = df.drop(columns=drop_features)
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    models = {
        'XGBoost': XGBClassifier(),
        'LightGBM': LGBMClassifier(),
        'CatBoost': CatBoostClassifier(verbose=0),
        'RandomForest': RandomForestClassifier(),
        'NeuralNetwork': MLPClassifier(max_iter=500),
        'LogisticRegression': LogisticRegression()
    }
    param_grids = {
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'max_depth': [-1, 10, 20],
            'learning_rate': [0.01, 0.1, 0.3],
            'num_leaves': [31, 50, 100],
            'feature_fraction': [0.8, 0.9, 1.0]
        },
        'CatBoost': {
            'iterations': [500, 1000, 1500],
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'l2_leaf_reg': [3, 5, 7],
            'bagging_temperature': [0.1, 0.5, 1.0]
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'NeuralNetwork': {
            'hidden_layer_sizes': [(128,), (128, 64)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'learning_rate': ['constant', 'adaptive']
        },
        'LogisticRegression': {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'saga']
        }
    }

    best_estimators = {}
    for model_name in models.keys():
        grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=5, scoring='roc_auc')
        grid_search.fit(X_train, y_train)
        best_estimators[model_name] = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    return best_estimators

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'AUROC': roc_auc_score(y_test, y_prob),
        'AUPRC': average_precision_score(y_test, y_prob)
    }
    
    return metrics

def ablation_study(data_path):
    df = load_data(data_path)

    # Full features
    X_train, X_val, y_train, y_val = prepare_data(df)
    models = train_model(X_train, y_train)
    full_results = {model_name: evaluate_model(model, X_val, y_val) for model_name, model in models.items()}

    # Without age
    X_train, X_val, y_train, y_val = prepare_data(df, drop_features=['age'])
    models = train_model(X_train, y_train)
    no_age_results = {model_name: evaluate_model(model, X_val, y_val) for model_name, model in models.items()}

    # Without sex
    X_train, X_val, y_train, y_val = prepare_data(df, drop_features=['sex'])
    models = train_model(X_train, y_train)
    no_sex_results = {model_name: evaluate_model(model, X_val, y_val) for model_name, model in models.items()}

    # Without age and sex
    X_train, X_val, y_train, y_val = prepare_data(df, drop_features=['age', 'sex'])
    models = train_model(X_train, y_train)
    no_age_sex_results = {model_name: evaluate_model(model, X_val, y_val) for model_name, model in models.items()}

    return full_results, no_age_results, no_sex_results, no_age_sex_results

def create_summary_table(full_results, no_age_results, no_sex_results, no_age_sex_results):
    summary_table = []
    for model_name in full_results.keys():
        summary_table.append({
            'Model': model_name,
            'AUROC (with all features)': full_results[model_name]['AUROC'],
            'AUROC (without age)': no_age_results[model_name]['AUROC'],
            'AUROC (without sex)': no_sex_results[model_name]['AUROC'],
            'AUROC (without age and sex)': no_age_sex_results[model_name]['AUROC']
        })
    return pd.DataFrame(summary_table)

if __name__ == "__main__":
    data_paths = {
        'Mortality Prediction': 'data/mortality_data.csv',
        'Readmission Prediction': 'data/readmission_data.csv',
        'PLOS Prediction': 'data/plos_data.csv'
    }
    
    all_results = {}
    for task, path in data_paths.items():
        print(f"Running ablation study for {task}...")
        full_results, no_age_results, no_sex_results, no_age_sex_results = ablation_study(path)
        summary_table = create_summary_table(full_results, no_age_results, no_sex_results, no_age_sex_results)
        summary_table.to_csv(f'results/{task}_ablation_summary.csv', index=False)
        all_results[task] = summary_table

    print("Ablation studies complete. Summary tables saved in the results directory.")
