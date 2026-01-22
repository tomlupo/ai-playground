"""Polish financial sentiment analysis using HerBERT."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import transformers, but make it optional for PoC
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers/torch not available. Using lexicon-only sentiment analysis.")


class PolishFinancialSentiment:
    """
    Sentiment analyzer for Polish financial text.

    Uses HerBERT model when available, falls back to lexicon-based analysis.
    """

    # Model version identifier
    MODEL_VERSION = "herbert-lexicon-v1"

    # Polish financial lexicons
    BULLISH_KEYWORDS = {
        # Buy signals
        "kupuj", "kup", "kupować", "akumuluj", "akumulować",
        # Positive trends
        "wzrost", "wzrosty", "rośnie", "rosną", "rosnący",
        "hossa", "hossy", "odbicie", "odbija",
        # Profits
        "zysk", "zyski", "zarabia", "zarabiać",
        # Technical breakouts
        "przebicie", "wybicie", "breakout",
        # Long positions
        "long", "longi", "trzymaj", "trzymać",
        # Targets
        "target", "cel", "docelowy",
        # Dividends
        "dywidenda", "dywidendy",
        # Recommendations
        "rekomendacja", "polecam", "warto",
        # Enthusiasm
        "super", "świetne", "bomba", "rakieta",
    }

    BEARISH_KEYWORDS = {
        # Sell signals
        "sprzedaj", "sprzedawaj", "redukuj", "unikaj",
        # Negative trends
        "spadek", "spadki", "spada", "spadają", "spadający",
        "bessa", "bessy", "korekta", "krach",
        # Losses
        "strata", "straty", "traci", "tracić",
        # Short positions
        "short", "shorty", "shortować",
        # Risks
        "ryzyko", "ryzykowne", "niebezpieczne",
        # Negative sentiment
        "bankructwo", "upadek", "dno",
        "dump", "zrzut", "manipulacja",
        # Warnings
        "uwaga", "ostrożnie", "bańka",
        # Disappointment
        "słabe", "kiepskie", "rozczarowanie",
    }

    NEUTRAL_MODIFIERS = {
        "może", "chyba", "prawdopodobnie", "możliwe",
        "nie wiem", "trudno powiedzieć", "zobaczymy",
    }

    NEGATION_WORDS = {
        "nie", "nigdy", "żaden", "brak", "bez",
    }

    def __init__(
        self,
        model_name: str = "allegro/herbert-base-cased",
        use_model: bool = True,
        lexicon_weight: float = 0.3,
    ):
        """
        Initialize sentiment analyzer.

        Args:
            model_name: HuggingFace model name for HerBERT
            use_model: Whether to use transformer model (if available)
            lexicon_weight: Weight for lexicon-based adjustment (0-1)
        """
        self.model_name = model_name
        self.lexicon_weight = lexicon_weight
        self.tokenizer: Optional[object] = None
        self.model: Optional[object] = None
        self.device = "cpu"

        if use_model and TRANSFORMERS_AVAILABLE:
            try:
                self._load_model()
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Using lexicon-only mode.")

    def _load_model(self) -> None:
        """Load the HerBERT model."""
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,  # positive, negative, neutral
        )
        self.model.eval()

        # Use GPU if available
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = self.model.to(self.device)
            logger.info("Using GPU for inference")

        PolishFinancialSentiment.MODEL_VERSION = f"herbert-{self.model_name.split('/')[-1]}"

    def analyze(self, text: str) -> tuple[float, str, float]:
        """
        Analyze sentiment of Polish financial text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (score, label, confidence)
            - score: -1.0 to 1.0
            - label: 'positive', 'negative', 'neutral'
            - confidence: 0.0 to 1.0
        """
        if not text or not text.strip():
            return 0.0, "neutral", 0.0

        # Get lexicon-based score
        lexicon_score = self._lexicon_sentiment(text.lower())

        # If model is available, combine scores
        if self.model is not None and self.tokenizer is not None:
            model_score, model_confidence = self._model_sentiment(text)
            # Weighted combination
            score = (1 - self.lexicon_weight) * model_score + self.lexicon_weight * lexicon_score
            confidence = model_confidence
        else:
            score = lexicon_score
            confidence = min(abs(lexicon_score) + 0.3, 1.0)

        # Clamp score
        score = max(-1.0, min(1.0, score))

        # Determine label
        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"

        return score, label, confidence

    def _lexicon_sentiment(self, text: str) -> float:
        """
        Calculate sentiment using lexicon approach.

        Args:
            text: Lowercase text to analyze

        Returns:
            Sentiment score from -1.0 to 1.0
        """
        words = text.split()
        if not words:
            return 0.0

        bullish_count = 0
        bearish_count = 0
        negation_active = False

        for i, word in enumerate(words):
            # Check for negation
            if word in self.NEGATION_WORDS:
                negation_active = True
                continue

            # Check bullish keywords
            if word in self.BULLISH_KEYWORDS or any(kw in word for kw in self.BULLISH_KEYWORDS):
                if negation_active:
                    bearish_count += 1
                else:
                    bullish_count += 1
                negation_active = False

            # Check bearish keywords
            elif word in self.BEARISH_KEYWORDS or any(kw in word for kw in self.BEARISH_KEYWORDS):
                if negation_active:
                    bullish_count += 1
                else:
                    bearish_count += 1
                negation_active = False

            # Reset negation after non-keyword word
            else:
                # Keep negation active for 2-3 words
                if i > 0 and words[i - 1] not in self.NEGATION_WORDS:
                    negation_active = False

        # Check for neutral modifiers
        neutral_count = sum(1 for mod in self.NEUTRAL_MODIFIERS if mod in text)

        # Calculate score
        total_keywords = bullish_count + bearish_count
        if total_keywords == 0:
            return 0.0

        raw_score = (bullish_count - bearish_count) / total_keywords

        # Apply neutral dampening
        if neutral_count > 0:
            raw_score *= 0.7

        return raw_score

    def _model_sentiment(self, text: str) -> tuple[float, float]:
        """
        Calculate sentiment using transformer model.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (score, confidence)
        """
        if self.model is None or self.tokenizer is None:
            return 0.0, 0.0

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # Convert to numpy
        probs = probs[0].cpu().numpy()

        # Assuming: 0=negative, 1=neutral, 2=positive
        score = float(probs[2] - probs[0])
        confidence = float(max(probs))

        return score, confidence

    def analyze_batch(self, texts: list[str], batch_size: int = 32) -> list[tuple[float, str, float]]:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for model inference

        Returns:
            List of (score, label, confidence) tuples
        """
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results

    @property
    def model_version(self) -> str:
        """Get model version string."""
        return PolishFinancialSentiment.MODEL_VERSION
