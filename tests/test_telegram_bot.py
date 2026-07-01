import unittest

try:
    from telegram_bot import safe_filename_part
except ModuleNotFoundError as error:
    if error.name != "telegram":
        raise
    raise unittest.SkipTest("python-telegram-bot is not installed")


class TelegramBotTests(unittest.TestCase):
    def test_safe_filename_part_removes_path_characters(self) -> None:
        self.assertEqual(safe_filename_part("../abc:123"), "___abc_123")

    def test_safe_filename_part_has_fallback(self) -> None:
        self.assertEqual(safe_filename_part("!!!"), "___")
        self.assertEqual(safe_filename_part(""), "photo")


if __name__ == "__main__":
    unittest.main()
