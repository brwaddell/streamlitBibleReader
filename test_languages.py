"""Language catalog used by the translator, audio generator, and localize script."""
import unittest

from languages import (
    LANGUAGE_CODES,
    TRANSLATION_TARGET_CODES,
    elevenlabs_language_code,
    language_label,
    translation_system_prompt,
)
from lib import elevenlabs_voice_male_for_language, male_narrator_label


class LanguageCatalogTests(unittest.TestCase):
    def test_includes_existing_and_new_languages(self):
        self.assertEqual(
            LANGUAGE_CODES,
            ["en", "es", "fr", "de", "pt", "it", "ar", "zh", "ja"],
        )

    def test_translation_targets_exclude_english(self):
        self.assertNotIn("en", TRANSLATION_TARGET_CODES)
        self.assertIn("es", TRANSLATION_TARGET_CODES)
        self.assertEqual(len(TRANSLATION_TARGET_CODES), 8)

    def test_elevenlabs_codes(self):
        self.assertEqual(elevenlabs_language_code("zh"), "zh")
        self.assertEqual(elevenlabs_language_code("pt"), "pt")
        self.assertEqual(elevenlabs_language_code("ja"), "ja")

    def test_male_narrator_is_johnny_kid_only_for_spanish(self):
        spanish = elevenlabs_voice_male_for_language("es")
        self.assertEqual(male_narrator_label("es"), "Johnny Kid")
        for code in ("en", "fr", "de", "pt", "it", "ar", "zh", "ja"):
            self.assertEqual(male_narrator_label(code), "Earl")
            self.assertNotEqual(elevenlabs_voice_male_for_language(code), spanish)

    def test_labels(self):
        self.assertEqual(language_label("fr"), "French (fr)")
        self.assertEqual(language_label("zh"), "Mandarin (zh)")

    def test_prompt_names_for_variants(self):
        mandarin = translation_system_prompt("zh", "Grade 2 (ages 5–7)")
        arabic = translation_system_prompt("ar", "Grade 2 (ages 5–7)")
        portuguese = translation_system_prompt("pt", "Grade 2 (ages 5–7)")
        self.assertIn("Simplified Chinese (Mandarin)", mandarin)
        self.assertIn("Modern Standard Arabic", arabic)
        self.assertIn("Brazilian Portuguese", portuguese)
        self.assertIn("Return only the translated text.", mandarin)


if __name__ == "__main__":
    unittest.main()
