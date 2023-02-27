import unittest

from finetoolformer.main import main

class TestFinetoolformer(unittest.TestCase):
    def test_main(self):
        response = main("Who is Thierry Henry?")

        self.assertTrue(
            response.startswith(
            " He is widely regarded as one of the greatest players of his generation"))