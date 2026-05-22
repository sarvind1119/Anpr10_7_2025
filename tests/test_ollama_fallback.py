import unittest

from app.ollama_fallback import score_indian_plate, select_best_indian_candidate


class OllamaFallbackTests(unittest.TestCase):
    def test_prefers_valid_indian_candidate(self):
        self.assertEqual(select_best_indian_candidate("UI12345", "UK07DX985"), "UK07DX985")

    def test_scores_indian_plate_higher_than_noise(self):
        self.assertGreater(score_indian_plate("MP09AB1234"), score_indian_plate("SSU7T1"))


if __name__ == "__main__":
    unittest.main()
