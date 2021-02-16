"""This script load data from a remote storage."""
import os
import gdown
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config
from loan_workflow.workflow import Workflow as wf

# Setting the random seed generator
np.random.seed(42)
random = Config.random_state

# Create file path to save the data after loading it.
Config.original.parent.mkdir(parents=True, exist_ok=True)
wf.create_file_path(Config.data)

# Download data from remote storage
gdown.download(
    os.environ.get("DATA"),
    str(Config.original)
)

# Loading the data obtained in a pandas DataFrame
#df = pd.read_excel(str(Config.original))
df = wf.load_df(str(Config.original))

# Splitting data into train and test sets
df_train, df_test = train_test_split(df, test_size=0.2, random_state=random)

# Saving splitted data to the data path created earlier
wf.dump_df(df_train, data='train.csv')
wf.dump_df(df_test, data='test.csv')