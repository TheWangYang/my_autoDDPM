from sklearn.metrics import roc_curve, roc_auc_score


def compute_auroc(y_pred, y):
    y_pred = y_pred.flatten()
    y = y.flatten()

    fpr, tpr, thresholds = roc_curve(y.astype(int), y_pred)
    auroc = roc_auc_score(y.astype(int), y_pred)
    return auroc, fpr, tpr, thresholds
