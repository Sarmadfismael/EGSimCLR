from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_precision(y_true, y_pred, average='macro'):
    return precision_score(y_true, y_pred, average=average)

def compute_recall(y_true, y_pred, average='macro'):
    return recall_score(y_true, y_pred, average=average)

def compute_f1(y_true, y_pred, average='macro'):
    return f1_score(y_true, y_pred, average=average)

def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

