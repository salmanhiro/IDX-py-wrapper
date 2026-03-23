"""Tests for the SentimentAnalyzer."""

from __future__ import annotations

from idx_wrapper.sentiment import SentimentAnalyzer


class TestSentimentAnalyzerScoreHeadline:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer()

    def test_positive_headline_returns_positive_score(self):
        score = self.analyzer.score_headline("Company reports record high profit and strong growth")
        assert score > 0

    def test_negative_headline_returns_negative_score(self):
        score = self.analyzer.score_headline("Bankruptcy and massive loss announced")
        assert score < 0

    def test_neutral_headline_near_zero(self):
        score = self.analyzer.score_headline("Company holds annual general meeting")
        assert -0.5 < score < 0.5

    def test_score_clamped_to_range(self):
        score = self.analyzer.score_headline(
            "Record high profit surge expansion dividend increase strong buy"
        )
        assert -1.0 <= score <= 1.0

    def test_empty_headline_returns_zero(self):
        assert self.analyzer.score_headline("") == 0.0


class TestSentimentAnalyzerAggregateScore:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer()

    def test_empty_list_returns_zero(self):
        assert self.analyzer.aggregate_score([]) == 0.0

    def test_mixed_headlines_return_mean(self):
        # positive + negative should average out
        score = self.analyzer.aggregate_score(
            ["strong profit growth", "loss and decline"]
        )
        assert -1.0 <= score <= 1.0

    def test_all_positive_headlines(self):
        score = self.analyzer.aggregate_score(
            ["revenue growth beats expectations", "strong dividend announced"]
        )
        assert score > 0


class TestSentimentAnalyzerLabel:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer()

    def test_very_positive(self):
        assert self.analyzer.label(0.8) == "Very Positive"

    def test_positive(self):
        assert self.analyzer.label(0.3) == "Positive"

    def test_neutral(self):
        assert self.analyzer.label(0.0) == "Neutral"

    def test_negative(self):
        assert self.analyzer.label(-0.3) == "Negative"

    def test_very_negative(self):
        assert self.analyzer.label(-0.8) == "Very Negative"
