import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from couple_match import box_blur, compare_photos, extract_features, format_report


class CoupleMatchTests(unittest.TestCase):
    def test_box_blur_preserves_shape(self) -> None:
        source = np.arange(25, dtype=np.float32).reshape(5, 5)

        blurred = box_blur(source, radius=1)

        self.assertEqual(blurred.shape, source.shape)

    def test_compare_similar_synthetic_faces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.jpg"
            second = Path(tmp) / "second.jpg"
            self._draw_face(first, background=(230, 210, 190), shift=-8)
            self._draw_face(second, background=(226, 208, 194), shift=8)

            report = compare_photos(first, second)

            self.assertEqual(report.mode, "face")
            self.assertGreaterEqual(report.score, 55)
            self.assertGreater(report.keypoint_matches, 0)
            self.assertTrue(report.title)
            self.assertTrue(report.verdict.endswith("."))
            self.assertIsNone(report.line_harmony)
            self.assertNotIn("Palm line harmony", format_report(report))

    def test_compare_similar_synthetic_palms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first_palm.jpg"
            second = Path(tmp) / "second_palm.jpg"
            self._draw_palm(first, shift=-4)
            self._draw_palm(second, shift=6)

            report = compare_photos(first, second, mode="palm")

            self.assertEqual(report.mode, "palm")
            self.assertGreaterEqual(report.score, 50)
            self.assertIsNotNone(report.line_harmony)
            self.assertGreater(report.line_harmony, 0.45)
            self.assertIn("palm", report.verdict.lower())
            self.assertIn("Palm line harmony", format_report(report))

    def test_feature_extraction_handles_plain_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            photo = Path(tmp) / "plain.jpg"
            Image.new("RGB", (128, 128), (120, 120, 120)).save(photo)

            features = extract_features(photo)

            self.assertEqual(features.descriptors.shape[0], 0)
            self.assertAlmostEqual(features.center_of_energy[0], 0.5)
            self.assertAlmostEqual(features.center_of_energy[1], 0.5)

    @staticmethod
    def _draw_face(path: Path, background: tuple[int, int, int], shift: int) -> None:
        image = Image.new("RGB", (420, 520), background)
        draw = ImageDraw.Draw(image)
        draw.ellipse((135 + shift, 95, 285 + shift, 245), fill=(235, 185, 150), outline=(85, 55, 45), width=4)
        draw.ellipse((172 + shift, 145, 187 + shift, 160), fill=(30, 30, 30))
        draw.ellipse((232 + shift, 145, 247 + shift, 160), fill=(30, 30, 30))
        draw.arc((180 + shift, 168, 240 + shift, 215), 20, 160, fill=(140, 50, 70), width=4)
        draw.rectangle((120 + shift, 270, 305 + shift, 500), fill=(50, 120, 210))
        draw.line((135 + shift, 105, 285 + shift, 245), fill=(255, 235, 210), width=3)
        image.save(path)

    @staticmethod
    def _draw_palm(path: Path, shift: int) -> None:
        image = Image.new("RGB", (420, 520), (236, 213, 190))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((115 + shift, 145, 305 + shift, 430), radius=75, fill=(229, 181, 148), outline=(120, 78, 62), width=3)
        for index, x in enumerate((130, 170, 210, 250, 290)):
            top = 50 + abs(index - 2) * 12
            draw.rounded_rectangle((x + shift, top, x + 36 + shift, 190), radius=18, fill=(231, 184, 151), outline=(120, 78, 62), width=3)
        draw.arc((145 + shift, 235, 300 + shift, 380), 190, 345, fill=(120, 70, 65), width=5)
        draw.arc((130 + shift, 190, 310 + shift, 345), 200, 325, fill=(135, 78, 74), width=4)
        draw.arc((150 + shift, 265, 270 + shift, 440), 205, 315, fill=(140, 82, 78), width=4)
        draw.line((165 + shift, 210, 245 + shift, 415), fill=(150, 88, 82), width=3)
        draw.line((255 + shift, 215, 205 + shift, 405), fill=(150, 88, 82), width=3)
        image.save(path)


if __name__ == "__main__":
    unittest.main()
