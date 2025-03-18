import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
import joblib

def generate_df_summary(model, y_test, y_pred, label, threshold=0.5):

    cm = confusion_matrix(y_test, y_pred,)
    cm_df = pd.DataFrame(cm, index=['True Class 0', 'True Class 1'], columns=['Pred Class 0', 'Pred Class 1'])

    precision = round(precision_score(y_test, y_pred),2)
    recall = round(recall_score(y_test, y_pred), 2)
    f1 = round(f1_score(y_test, y_pred), 2)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print('\nConfusion Matrix:\n\n', cm_df)

    model_params = model.get_params()

    performance_dict = {}
    for i,j in zip([label, datetime.today().date(), precision, recall, f1], ['Name', 'Date', 'Precision', 'Recall', 'F1-Score']):
        performance_dict[j] = i

    for i,j in zip(cm.flatten(), ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']):
        performance_dict[j] = i

    performance_dict.update(model_params)
    performance_dict['proba_threshold'] = threshold

    file_path = 'output/model_performance.csv'
    file_exists = os.path.isfile(file_path)
    df = pd.DataFrame([performance_dict])

    df.to_csv(file_path, mode='a', header=not file_exists, index=False)


def predict_max_f1(model, X_test, y_test):
    # get model probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # precision recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    precision, recall = precision[:-1], recall[:-1]

    # get f1 scores
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Small value to prevent division by zero

    # find threshold that maximises f1
    best_threshold = thresholds[np.argmax(f1_scores)]

    # apply best threshold to yproba
    y_pred_adjusted = (y_proba >= best_threshold).astype(int)

    return y_pred_adjusted, best_threshold

def save_model(search, model_name):
    os.makedirs(f'saved_models/{model_name}', exist_ok=True)

    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values('rank_test_score').reset_index(drop=True)
    results_df.to_csv(f'saved_models/{model_name}/results.csv')
    joblib.dump(search.best_estimator_, f"saved_models/{model_name}/best_model.pkl")
    print('Random search results and best model saved.')