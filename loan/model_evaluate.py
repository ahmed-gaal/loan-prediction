"""This script evaluates the accuracy of the model trained."""
import numpy as np
from config import Config
from pipeline import Workflow
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score

# Create a file directory to save the metrics of the model.
Workflow.create_file_path(Config.metrics)

# Load test data to commence model evaluation.
X_test = Workflow.load_features(data='test_features.csv')
y_test = Workflow.load_features(data='test_target.csv')

# Load the model to commence predictions.
clf = Workflow.load_pickle(filename='model.pickle', func='rb')

# Performing predictions on the loaded model.
y_pred = clf.predict(X_test)

# Calculating metrics for the model based on the predictions.
prec = precision_score(y_test, y_pred)
prec_weight = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred)
rec_weight = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred)
f1_weight = f1_score(y_test, y_pred)
accu = np.mean(accuracy_score(y_test, y_pred))
std = np.std(accuracy_score(y_test, y_pred))

# Storing metrics in models directory.
Workflow.dump_json(
    accu, std, prec, prec_weight, rec, rec_weight, f1, f1_weight
)
Workflow.confusion_matrix(y_test, y_pred)