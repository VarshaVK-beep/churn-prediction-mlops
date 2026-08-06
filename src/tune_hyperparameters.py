import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print('Best F1 score:', study.best_value)
print('Best params:', study.best_params)

# Train final model with best params and save
best_model = XGBClassifier(**study.best_params, eval_metric='logloss', random_state=42)
best_model.fit(X_train, y_train)

import joblib
joblib.dump(best_model, 'src/best_model.pkl')
print('Best model saved to src/best_model.pkl')
