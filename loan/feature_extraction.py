"""This is a script to perform feature extraction and preprocessing."""
import pandas as pd
from loan_workflow import Workflow
from config import Config

wf = Workflow()
# Create a file directory to store features extracted.
wf.create_file_path(Config.features)

# Loading our data into a pandas DataFrame.
train_df = wf.load_df(Config.data, data='train.csv')
test_df = wf.load_df(Config.data, data='test.csv')

# Perform Feature Extraction and Preprocessing
train_features = wf.feature_extraction(train_df)
test_features = wf.feature_extraction(test_df)

# Saving the preprocessed features to features directory in pandas DataFrame
pd.DataFrame(
    train_features, columns=[
        'PaidMin', 'PropertyArea', 'ApplicantIncome', 'PaidHour', 'Purpose',
        'LoanAmount', 'Gender'
    ]
).to_csv(str(Config.features / 'train_features.csv'), index=None)
pd.DataFrame(
    test_features, columns=[
        'PaidMin', 'PropertyArea', 'ApplicantIncome', 'PaidHour', 'Purpose',
        'LoanAmount', 'Gender'
    ]
).to_csv(str(Config.features / 'test_features.csv'), index=None)


# Extract target from the data and store in features
train_target = train_df['LoanStatus'].to_csv(
    str(Config.features / 'train_target.csv'), index=None
)
test_target = test_df['LoanStatus'].to_csv(
    str(Config.features / 'test_target.csv'), index=None
)