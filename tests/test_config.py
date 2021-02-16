import unittest
from loan.config import Config
from pathlib import Path


class TestConfig(unittest.TestCase):
    """Instantiate the class"""

    def test_random_state(self):
        con = Config()
        self.assertEqual(con.random_state, 42)
    

    def test_assets_path(self):
        con = Config()
        self.assertEqual(con.assets_path, Path('./loan_assets'))

    
    def test_original_data_path(self):
        con = Config()
        self.assertEqual(
            con.original, Path('./loan_assets/loan_original/loan.csv')
        )

    
    def test_splitted_dataset_path(self):
        con = Config()
        self.assertEqual(con.data, Path('./loan_assets/loan_data'))

    
    def test_features_path(self):
        con = Config()
        self.assertEqual(con.features, Path('./loan_assets/loan_features'))

    
    def test_models_path(self):
        con = Config()
        self.assertEqual(con.models, Path('./loan_assets/loan_models'))


    def test_metrics_path(self):
        con = Config()
        self.assertEqual(con.metrics, Path('./loan_assets/loan_metrics.json'))