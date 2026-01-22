"""Tests for sentiment analysis module."""

import pytest

from src.nlp.sentiment import PolishFinancialSentiment


class TestPolishFinancialSentiment:
    """Tests for PolishFinancialSentiment class."""

    @pytest.fixture
    def analyzer(self):
        """Create sentiment analyzer instance."""
        return PolishFinancialSentiment(use_model=False)

    def test_bullish_text(self, analyzer):
        """Test that bullish text returns positive sentiment."""
        text = "Kupuj CDR! Wzrosty są pewne, target 500 PLN"
        score, label, confidence = analyzer.analyze(text)

        assert score > 0, f"Expected positive score, got {score}"
        assert label == "positive"
        assert 0 <= confidence <= 1

    def test_bearish_text(self, analyzer):
        """Test that bearish text returns negative sentiment."""
        text = "Sprzedaj wszystko! Spadki będą kontynuowane, krach nieunikniony"
        score, label, confidence = analyzer.analyze(text)

        assert score < 0, f"Expected negative score, got {score}"
        assert label == "negative"
        assert 0 <= confidence <= 1

    def test_neutral_text(self, analyzer):
        """Test that neutral text returns neutral sentiment."""
        text = "Zobaczymy co będzie jutro na giełdzie"
        score, label, confidence = analyzer.analyze(text)

        assert -0.3 <= score <= 0.3, f"Expected near-zero score, got {score}"
        assert label in ["neutral", "positive", "negative"]

    def test_empty_text(self, analyzer):
        """Test that empty text returns neutral with zero confidence."""
        score, label, confidence = analyzer.analyze("")

        assert score == 0
        assert label == "neutral"
        assert confidence == 0

    def test_negation_handling(self, analyzer):
        """Test that negation is handled correctly."""
        # "nie kupuj" should be bearish
        text_negated = "Nie kupuj tych akcji"
        score_neg, _, _ = analyzer.analyze(text_negated)

        # "kupuj" should be bullish
        text_positive = "Kupuj te akcje"
        score_pos, _, _ = analyzer.analyze(text_positive)

        assert score_neg < score_pos, "Negation should reduce sentiment score"

    def test_batch_analysis(self, analyzer):
        """Test batch analysis."""
        texts = [
            "Kupuj! Wzrosty!",
            "Sprzedaj! Spadki!",
            "Zobaczymy",
        ]

        results = analyzer.analyze_batch(texts)

        assert len(results) == 3
        assert results[0][0] > 0  # First should be positive
        assert results[1][0] < 0  # Second should be negative

    def test_model_version(self, analyzer):
        """Test that model version is set."""
        assert analyzer.model_version is not None
        assert len(analyzer.model_version) > 0
