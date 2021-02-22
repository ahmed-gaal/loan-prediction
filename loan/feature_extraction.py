"""This is a script to perform feature extraction and preprocessing."""
import pandas as pd
from pipeline import Workflow
from config import Config

# Create a file directory to store features extracted.
Workflow.create_file_path(Config.features)

# Loading our data into a pandas DataFrame.
train_df = Workflow.load_df(Config.data, data='train.csv')
test_df = Workflow.load_df(Config.data, data='test.csv')

# Perform Feature Extraction and Preprocessing
train_features = Workflow.feature_extraction(train_df, scale=True)
test_features = Workflow.feature_extraction(test_df, scale=False)

# Saving the preprocessed features to features directory in pandas DataFrame
pd.DataFrame(
    train_features
).to_csv(str(Config.features / 'train_features.csv'), index=None)
pd.DataFrame(
    test_features
).to_csv(str(Config.features / 'test_features.csv'), index=None)


# Extract target from the data and store in features
train_target = train_df['LoanStatus'].to_csv(
    str(Config.features / 'train_target.csv'), index=None
)
test_target = test_df['LoanStatus'].to_csv(
    str(Config.features / 'test_target.csv'), index=None
)
