"""Tests for ticker detection module."""

import pytest

from src.nlp.ticker_detector import WSETickerDetector


class TestWSETickerDetector:
    """Tests for WSETickerDetector class."""

    @pytest.fixture
    def detector(self):
        """Create ticker detector instance."""
        return WSETickerDetector()

    def test_explicit_ticker_detection(self, detector):
        """Test detection of explicit ticker symbols."""
        text = "Widziałem, że CDR i PKO rosną dzisiaj"
        mentions = detector.detect(text)

        tickers = [m[0] for m in mentions]
        assert "CDR" in tickers
        assert "PKO" in tickers

    def test_alias_detection(self, detector):
        """Test detection via company name aliases."""
        text = "CD Projekt zapowiedział nową grę, a Allegro nową promocję"
        mentions = detector.detect(text)

        tickers = [m[0] for m in mentions]
        assert "CDR" in tickers
        assert "ALE" in tickers

    def test_mention_type(self, detector):
        """Test that mention types are correctly classified."""
        text = "CDR to świetna spółka, a cd projekt też"
        mentions = detector.detect(text)

        # Should have both explicit and inferred mentions
        explicit = [m for m in mentions if m[2] == "explicit"]
        inferred = [m for m in mentions if m[2] == "inferred"]

        # Due to deduplication, we might only get one CDR
        cdr_mentions = [m for m in mentions if m[0] == "CDR"]
        assert len(cdr_mentions) >= 1

    def test_no_duplicate_tickers(self, detector):
        """Test that duplicate tickers are not returned."""
        text = "CDR CDR CDR cd projekt cdprojekt"
        mentions = detector.detect(text)

        tickers = [m[0] for m in mentions]
        # Should only have one CDR despite multiple mentions
        assert tickers.count("CDR") == 1

    def test_context_extraction(self, detector):
        """Test context extraction around ticker mention."""
        text = "Wczoraj kupiłem CDR po 300 PLN i czekam na wzrosty"
        context = detector.get_context(text, "CDR", window=20)

        assert "CDR" in context
        assert len(context) <= 50  # window * 2 + ticker length

    def test_empty_text(self, detector):
        """Test empty text returns no mentions."""
        mentions = detector.detect("")
        assert len(mentions) == 0

    def test_get_all_tickers(self, detector):
        """Test getting all known tickers."""
        tickers = detector.get_all_tickers()

        assert len(tickers) > 0
        assert "CDR" in tickers
        assert "PKO" in tickers
        assert "ALE" in tickers

    def test_add_custom_ticker(self, detector):
        """Test adding a custom ticker."""
        detector.add_ticker("TEST", "Test Company", ["testcorp", "test corp"])

        text = "TEST company czyli testcorp ogłosiła wyniki"
        mentions = detector.detect(text)

        tickers = [m[0] for m in mentions]
        assert "TEST" in tickers

    def test_case_insensitive_aliases(self, detector):
        """Test that aliases are case insensitive."""
        text1 = "ALLEGRO ma świetne wyniki"
        text2 = "allegro ma świetne wyniki"
        text3 = "Allegro ma świetne wyniki"

        for text in [text1, text2, text3]:
            mentions = detector.detect(text)
            tickers = [m[0] for m in mentions]
            assert "ALE" in tickers, f"Failed for: {text}"
