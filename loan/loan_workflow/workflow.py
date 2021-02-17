"""This script stores all methods using in this workflow."""
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from config import Config


class Workflow():
    """Create a workflow object"""

    
    def __init__(self):
        """Instantiate the object"""


    def create_file_path(self, path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        return

    
    def load_original_df(path):
        """Load dataset into a pandas DataFrame"""
        df = pd.read_csv(path, engine='python')
        return df

    def load_df(self, path, data=None):
        df = pd.read_csv(str(path / data))
        return df


    def dump_df(dframe=None, name=None):
        """Method to save a pandas DataFrame in a csv file"""
        dframe.to_csv(str(Config.data / name), index=None)

    
    def feature_extraction(self, df):
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


