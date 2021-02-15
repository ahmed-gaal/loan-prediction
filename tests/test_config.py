import unittest


class TestConfig(unittest.TestCase):
    """Instantiate the class"""

    def test_random_state(self):
        con = Config()
        self.assertEqual(con.random_state, 42)