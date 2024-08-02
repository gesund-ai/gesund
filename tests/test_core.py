import unittest
from gesund.core import GesundSDK

class TestGesundSDK(unittest.TestCase):
    def test_connect(self):
        sdk = GesundSDK(api_key="dummy_key")
        sdk.connect()
        # Add assertions here

    def test_get_data(self):
        sdk = GesundSDK(api_key="dummy_key")
        data = sdk.get_data()
        self.assertEqual(data, {})

if __name__ == '__main__':
    unittest.main()
