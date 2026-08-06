import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

mlflow.set_experiment('churn-prediction')

def log_model_run(name, model, is_xgboost=False):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param('model_type', name)
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('roc_auc', auc)

        if is_xgboost:
            mlflow.xgboost.log_model(model, 'model')
        else:
            mlflow.sklearn.log_model(model, 'model')

        print(f'{name} -> Accuracy: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}')

log_model_run('LogisticRegression', LogisticRegression(max_iter=1000))
log_model_run('XGBoost', XGBClassifier(eval_metric='logloss', random_state=42), is_xgboost=True)

print('\nRun: mlflow ui  --> then open http://localhost:5000 to view results')
