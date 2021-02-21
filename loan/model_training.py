"""This script performs model selection and model training."""
from sklearn.linear_model import SGDClassifier
from config import Config
from pipeline import Workflow


# Create a file directory to save models trained.
Workflow.create_file_path(path=Config.models)

# Load features to commence training
X_train = Workflow.load_df(Config.features, data='train_features.csv')
y_train = Workflow.load_df(Config.features, data='train_target.csv')

# Instantiating model algorithm
clf = SGDClassifier(alpha=2.558105, l1_ratio=0.86)

# Fitting features to the algorithm
clf.fit(X_train, y_train.to_numpy().ravel())

# Store the model trained to models directory
Workflow.dump_pickle(clf, filename='model.pickle', func='wb')