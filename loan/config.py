from pathlib import Path


class Config:
    random_state=42
    assets_path = Path('./loan_assets')
    original = assets_path / 'loan_original' / 'clean_loan.csv'
    data = assets_path / 'loan_data'
    features = assets_path / 'loan_features'
    models = assets_path / 'loan_models'
    metrics =  assets_path / 'loan_metrics'