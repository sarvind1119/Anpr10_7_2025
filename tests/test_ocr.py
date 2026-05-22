import unittest

from app.ocr import normalize_plate_text


class NormalizePlateTextTests(unittest.TestCase):
    def test_normalizes_whitespace_and_hyphen(self):
        self.assertEqual(normalize_plate_text("mp 09-ab 1234"), "MP09AB1234")

    def test_removes_non_alphanumeric_characters(self):
        self.assertEqual(normalize_plate_text(" DL-3C$AF*0987 "), "DL3CAF0987")


if __name__ == "__main__":
    unittest.main()
