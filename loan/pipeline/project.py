"""This script stores all methods using in this workflow."""
import pickle
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from config import Config


class Workflow():
    """Create a workflow object"""

    
    def __init__(self):
        """Instantiate the object"""


    def create_file_path(path):
        path.mkdir(parents=True, exist_ok=True)
        return

    
    def load_original_df(path):
        """Load dataset into a pandas DataFrame"""
        df = pd.read_csv(path, engine='python')
        return df

    def load_df(path, data=None):
        df = pd.read_csv(str(path / data))
        return df


    def dump_df(dframe=None, name=None):
        """Method to save a pandas DataFrame in a csv file"""
        dframe.to_csv(str(Config.data / name), index=None)

    
    def feature_extraction(df):
        """
        Method for extracting and preprocessing features from
        the given dataset.
        """
        ft = df[['PaidMin', 'PropertyArea', 'ApplicantIncome', 'PaidHour',
                 'Purpose', 'LoanAmount', 'Gender']]
        # Scaling the features
        scale = StandardScaler()
        # Fitting to the train and test sets.
        scaled = scale.fit_transform(ft)
        return scaled


    def load_features(data=None):
        """Method to load features in a dataframe"""
        df = pd.read_csv(str(Config.features / data))
        return df

    
    def load_pickle(filename=None, func=None):
        """"Method to load pickle file."""
        return pickle.load(open(
            str(Config.models / filename), func
        ))

    
    def dump_pickle(model, filename=None, func=None):
        """Method to store model to pickle file."""
        return pickle.dump(model, open(
            str(Config.models / filename), func
        ))


    def dump_json(acc, std_acc, pre, pre1, re, re1, f1, f2):
        """Method to store metrics in a json file."""
        with open(str(Config.metrics / 'metrics.json'), 'w') as outfile:
            json.dump(
                (dict(precision=pre, weighted=pre1),
                 dict(recall=re, weighted=re1),
                 dict(f1_score=f1, weighted=f2),
                 dict(average_accuracy=acc, accuracy_std=std_acc)),
                 outfile
            )
    

    def confusion_matrix(truth, predictions):
        """Method to calculate the confusion matrix of the model."""
        con = pd.DataFrame(
            confusion_matrix(truth, predictions),
            index=pd.MultiIndex.from_product(
                [['Actual'], ['Negative', 'Positive']]
            ),
            columns=pd.MultiIndex.from_product(
                [['Predicted'], ['Negative', 'Positive']]
            )
        )

        con.to_csv(
            str(Config.metrics / 'confusion_matrix.csv')
        )

