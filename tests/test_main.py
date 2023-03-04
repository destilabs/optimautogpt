import unittest

from finetoolformer.main import pipeline

class TestFinetoolformer(unittest.TestCase):
    def test_main(self):
        response = pipeline("Who is Thierry Henry?")

        self.assertTrue("football" in response)