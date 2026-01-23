"""
Machine Learning pipeline for quantitative research.

Provides ML models for:
- Return prediction (regression)
- Direction classification
- Feature engineering from technical indicators
- Model evaluation and selection
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Literal
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


@dataclass
class MLResult:
    """Container for ML model results."""

    model_name: str
    model_type: Literal["regression", "classification"]
    train_score: float
    test_score: float
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    predictions: np.ndarray
    feature_importance: dict[str, float] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    model: Any = None


class FeatureEngineer:
    """Engineer features from OHLCV data for ML models."""

    def __init__(self, lookback_periods: list[int] | None = None):
        self.lookback_periods = lookback_periods or [5, 10, 20, 60]

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create ML features from OHLCV data.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else None

        features = pd.DataFrame(index=df.index)

        # Returns at different horizons
        for period in self.lookback_periods:
            features[f"return_{period}d"] = close.pct_change(period)
            features[f"volatility_{period}d"] = close.pct_change().rolling(period).std()

        # Price momentum
        for period in self.lookback_periods:
            features[f"momentum_{period}d"] = close / close.shift(period) - 1

        # Moving average features
        for period in self.lookback_periods:
            sma = close.rolling(period).mean()
            features[f"sma_ratio_{period}d"] = close / sma - 1
            features[f"sma_slope_{period}d"] = sma.pct_change(5)

        # RSI
        for period in [14, 28]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            features[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # Bollinger Band position
        for period in [20]:
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f"bb_position_{period}d"] = (close - sma) / (2 * std)

        # High-Low range
        for period in self.lookback_periods:
            features[f"hl_range_{period}d"] = (
                high.rolling(period).max() - low.rolling(period).min()
            ) / close

        # Volume features (if available)
        if volume is not None:
            for period in self.lookback_periods:
                vol_sma = volume.rolling(period).mean()
                features[f"volume_ratio_{period}d"] = volume / vol_sma
                features[f"volume_trend_{period}d"] = vol_sma.pct_change(period)

        # MACD features
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features["macd"] = macd / close  # Normalized
        features["macd_signal"] = signal / close
        features["macd_histogram"] = (macd - signal) / close

        # ATR (Average True Range)
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        for period in [14, 28]:
            features[f"atr_{period}d"] = tr.rolling(period).mean() / close

        # Day of week (cyclical encoding)
        if hasattr(df.index, "dayofweek"):
            features["day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
            features["day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)

        return features

    def create_target(
        self,
        data: pd.DataFrame,
        horizon: int = 1,
        target_type: Literal["return", "direction", "volatility"] = "return",
    ) -> pd.Series:
        """
        Create target variable for ML models.

        Args:
            data: DataFrame with OHLCV columns
            horizon: Prediction horizon in days
            target_type: Type of target variable

        Returns:
            Series with target values
        """
        close = data["close"]

        if target_type == "return":
            return close.pct_change(horizon).shift(-horizon)
        elif target_type == "direction":
            returns = close.pct_change(horizon).shift(-horizon)
            return (returns > 0).astype(int)
        elif target_type == "volatility":
            return close.pct_change().rolling(horizon).std().shift(-horizon)
        else:
            raise ValueError(f"Unknown target type: {target_type}")


class MLPipeline:
    """
    Machine Learning pipeline for financial prediction.

    Supports both regression and classification tasks with
    proper time-series cross-validation.
    """

    REGRESSION_MODELS = {
        "ridge": Ridge,
        "lasso": Lasso,
        "elastic_net": ElasticNet,
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "svr": SVR,
    }

    CLASSIFICATION_MODELS = {
        "logistic": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "svc": SVC,
    }

    def __init__(
        self,
        task: Literal["regression", "classification"] = "regression",
        n_splits: int = 5,
        scaler: Literal["standard", "robust"] = "standard",
    ):
        self.task = task
        self.n_splits = n_splits
        self.scaler_type = scaler
        self.feature_engineer = FeatureEngineer()

        self.scaler = StandardScaler() if scaler == "standard" else RobustScaler()
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

        self.models = (
            self.REGRESSION_MODELS if task == "regression" else self.CLASSIFICATION_MODELS
        )

    def prepare_data(
        self,
        data: pd.DataFrame,
        horizon: int = 1,
        target_type: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for ML.

        Args:
            data: OHLCV DataFrame
            horizon: Prediction horizon
            target_type: Target type (auto-detected if None)

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if target_type is None:
            target_type = "return" if self.task == "regression" else "direction"

        features = self.feature_engineer.create_features(data)
        target = self.feature_engineer.create_target(data, horizon, target_type)

        # Align and drop NaN
        combined = pd.concat([features, target.rename("target")], axis=1).dropna()
        X = combined.drop("target", axis=1)
        y = combined["target"]

        return X, y

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        **model_params,
    ) -> MLResult:
        """
        Train a single model with cross-validation.

        Args:
            X: Feature DataFrame
            y: Target Series
            model_name: Name of model to train
            **model_params: Additional model parameters

        Returns:
            MLResult with training results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        # Create pipeline with scaling
        model_class = self.models[model_name]

        # Set default params for certain models
        default_params = {}
        if model_name == "logistic":
            default_params = {"max_iter": 1000, "random_state": 42}
        elif model_name in ["random_forest", "gradient_boosting"]:
            default_params = {"random_state": 42, "n_estimators": 100}
        elif model_name == "svr":
            default_params = {"kernel": "rbf"}
        elif model_name == "svc":
            default_params = {"kernel": "rbf", "random_state": 42}

        default_params.update(model_params)
        model = model_class(**default_params)

        pipe = Pipeline([("scaler", self.scaler), ("model", model)])

        # Time series cross-validation
        scoring = "neg_mean_squared_error" if self.task == "regression" else "accuracy"
        cv_scores = cross_val_score(
            pipe, X, y, cv=self.tscv, scoring=scoring, error_score="raise"
        )

        if self.task == "regression":
            cv_scores = -cv_scores  # Convert back to positive MSE

        # Train on full data
        pipe.fit(X, y)
        predictions = pipe.predict(X)

        # Calculate metrics
        if self.task == "regression":
            train_score = r2_score(y, predictions)
            metrics = {
                "mse": mean_squared_error(y, predictions),
                "rmse": np.sqrt(mean_squared_error(y, predictions)),
                "mae": mean_absolute_error(y, predictions),
                "r2": train_score,
            }
        else:
            train_score = accuracy_score(y, predictions)
            metrics = {
                "accuracy": train_score,
                "precision": precision_score(y, predictions, zero_division=0),
                "recall": recall_score(y, predictions, zero_division=0),
                "f1": f1_score(y, predictions, zero_division=0),
            }

        # Feature importance (if available)
        feature_importance = None
        if hasattr(pipe.named_steps["model"], "feature_importances_"):
            importance = pipe.named_steps["model"].feature_importances_
            feature_importance = dict(zip(X.columns, importance))
        elif hasattr(pipe.named_steps["model"], "coef_"):
            coef = pipe.named_steps["model"].coef_
            if coef.ndim > 1:
                coef = coef[0]
            feature_importance = dict(zip(X.columns, np.abs(coef)))

        return MLResult(
            model_name=model_name,
            model_type=self.task,
            train_score=train_score,
            test_score=cv_scores.mean(),
            cv_scores=cv_scores,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            predictions=predictions,
            feature_importance=feature_importance,
            metrics=metrics,
            model=pipe,
        )

    def train_all_models(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, MLResult]:
        """
        Train all available models and compare results.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Dictionary of model results
        """
        results = {}
        for model_name in self.models:
            try:
                results[model_name] = self.train_model(X, y, model_name)
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        return results

    def grid_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        param_grid: dict,
    ) -> tuple[Any, dict]:
        """
        Perform grid search for hyperparameter tuning.

        Args:
            X: Feature DataFrame
            y: Target Series
            model_name: Model to tune
            param_grid: Parameter grid for search

        Returns:
            Tuple of (best model, best parameters)
        """
        model_class = self.models[model_name]

        # Adjust param names for pipeline
        adjusted_grid = {f"model__{k}": v for k, v in param_grid.items()}

        pipe = Pipeline([
            ("scaler", self.scaler),
            ("model", model_class()),
        ])

        scoring = "neg_mean_squared_error" if self.task == "regression" else "accuracy"
        grid_search = GridSearchCV(
            pipe, adjusted_grid, cv=self.tscv, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X, y)

        # Extract best params without pipeline prefix
        best_params = {
            k.replace("model__", ""): v for k, v in grid_search.best_params_.items()
        }

        return grid_search.best_estimator_, best_params

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using trained model."""
        return model.predict(X)

    def compare_models(self, results: dict[str, MLResult]) -> pd.DataFrame:
        """
        Create comparison DataFrame of all model results.

        Args:
            results: Dictionary of MLResult objects

        Returns:
            DataFrame comparing model performance
        """
        comparison_data = []
        for name, result in results.items():
            row = {
                "Model": name,
                "CV Mean": result.cv_mean,
                "CV Std": result.cv_std,
                "Train Score": result.train_score,
            }
            row.update(result.metrics)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by appropriate metric
        sort_col = "CV Mean" if self.task == "regression" else "accuracy"
        ascending = self.task == "regression"  # Lower MSE is better
        df = df.sort_values(sort_col, ascending=ascending)

        return df

    def get_top_features(
        self, result: MLResult, n: int = 10
    ) -> list[tuple[str, float]]:
        """Get top N most important features."""
        if result.feature_importance is None:
            return []

        sorted_features = sorted(
            result.feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_features[:n]


def run_ml_analysis(
    data: pd.DataFrame,
    task: Literal["regression", "classification"] = "classification",
    horizon: int = 5,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run complete ML analysis on price data.

    Args:
        data: OHLCV DataFrame
        task: ML task type
        horizon: Prediction horizon
        verbose: Print progress

    Returns:
        Dictionary with all results
    """
    if verbose:
        print(f"Running ML analysis ({task}, horizon={horizon}d)")

    pipeline = MLPipeline(task=task)

    # Prepare data
    if verbose:
        print("  Preparing features...")
    X, y = pipeline.prepare_data(data, horizon=horizon)
    if verbose:
        print(f"  Features: {X.shape[1]}, Samples: {len(X)}")

    # Train all models
    if verbose:
        print("  Training models...")
    results = pipeline.train_all_models(X, y)

    # Compare models
    comparison = pipeline.compare_models(results)
    if verbose:
        print("\nModel Comparison:")
        print(comparison.to_string(index=False))

    # Get best model
    if task == "regression":
        best_name = comparison.iloc[0]["Model"]  # Lowest CV MSE
    else:
        best_name = comparison.iloc[-1]["Model"]  # Highest accuracy

    best_result = results[best_name]

    if verbose:
        print(f"\nBest Model: {best_name}")
        if best_result.feature_importance:
            print("\nTop Features:")
            for feat, imp in pipeline.get_top_features(best_result, 5):
                print(f"  {feat}: {imp:.4f}")

    return {
        "pipeline": pipeline,
        "features": X,
        "target": y,
        "results": results,
        "comparison": comparison,
        "best_model": best_name,
        "best_result": best_result,
    }
