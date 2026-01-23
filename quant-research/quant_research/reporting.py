"""
HTML Report Generation module.

Generates interactive HTML reports with:
- Plotly visualizations
- Pandas DataFrames as tables
- Navigation menus
- Professional styling

Inspired by qreporting (https://github.com/tomlupo/qreporting)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ReportBuilder:
    """
    Build interactive HTML reports from nested data structures.

    Supports:
    - DataFrames (rendered as tables)
    - Plotly figures
    - Dictionaries (rendered as key-value tables)
    - Strings and numbers (rendered as text)
    """

    def __init__(self, title: str = "Quant Research Report"):
        self.title = title
        self.sections: dict[str, dict[str, Any]] = {}
        self.created_at = datetime.now()

    def add_section(self, name: str) -> None:
        """Add a new section to the report."""
        if name not in self.sections:
            self.sections[name] = {}

    def add_content(self, section: str, name: str, content: Any) -> None:
        """
        Add content to a section.

        Args:
            section: Section name
            name: Content name/title
            content: Content (DataFrame, Figure, dict, str, etc.)
        """
        if section not in self.sections:
            self.add_section(section)
        self.sections[section][name] = content

    def _render_dataframe(self, df: pd.DataFrame, name: str) -> str:
        """Render DataFrame as HTML table."""
        # Format numeric columns
        formatted_df = df.copy()
        for col in formatted_df.select_dtypes(include=[np.number]).columns:
            if formatted_df[col].abs().max() < 1:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
            elif formatted_df[col].abs().max() < 100:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}")
            else:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.0f}")

        table_html = formatted_df.to_html(
            classes="data-table",
            index=True,
            border=0,
            escape=False,
        )

        return f"""
        <div class="content-block">
            <h3>{name}</h3>
            <div class="table-container">
                {table_html}
            </div>
        </div>
        """

    def _render_figure(self, fig: go.Figure, name: str) -> str:
        """Render Plotly figure as HTML."""
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
        return f"""
        <div class="content-block">
            <h3>{name}</h3>
            <div class="figure-container">
                {fig_html}
            </div>
        </div>
        """

    def _render_dict(self, data: dict, name: str) -> str:
        """Render dictionary as key-value table."""
        rows = []
        for key, value in data.items():
            if isinstance(value, float):
                if abs(value) < 1:
                    value_str = f"{value:.4f}"
                elif abs(value) < 100:
                    value_str = f"{value:.2f}"
                else:
                    value_str = f"{value:,.0f}"
            else:
                value_str = str(value)
            rows.append(f"<tr><td class='key'>{key}</td><td class='value'>{value_str}</td></tr>")

        return f"""
        <div class="content-block">
            <h3>{name}</h3>
            <table class="kv-table">
                {''.join(rows)}
            </table>
        </div>
        """

    def _render_text(self, text: str, name: str) -> str:
        """Render text content."""
        return f"""
        <div class="content-block">
            <h3>{name}</h3>
            <p class="text-content">{text}</p>
        </div>
        """

    def _render_content(self, name: str, content: Any) -> str:
        """Render any content type to HTML."""
        if isinstance(content, pd.DataFrame):
            return self._render_dataframe(content, name)
        elif isinstance(content, go.Figure):
            return self._render_figure(content, name)
        elif isinstance(content, dict):
            return self._render_dict(content, name)
        else:
            return self._render_text(str(content), name)

    def _get_styles(self) -> str:
        """Get CSS styles for the report."""
        return """
        <style>
            :root {
                --primary-color: #1a73e8;
                --secondary-color: #5f6368;
                --bg-color: #f8f9fa;
                --card-bg: #ffffff;
                --border-color: #dadce0;
                --text-color: #202124;
                --success-color: #34a853;
                --warning-color: #fbbc04;
                --error-color: #ea4335;
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                line-height: 1.6;
            }

            .container {
                display: flex;
                min-height: 100vh;
            }

            /* Sidebar Navigation */
            .sidebar {
                width: 280px;
                background-color: var(--card-bg);
                border-right: 1px solid var(--border-color);
                padding: 20px;
                position: fixed;
                height: 100vh;
                overflow-y: auto;
            }

            .sidebar h1 {
                font-size: 1.4rem;
                color: var(--primary-color);
                margin-bottom: 10px;
                padding-bottom: 10px;
                border-bottom: 2px solid var(--primary-color);
            }

            .sidebar .timestamp {
                font-size: 0.8rem;
                color: var(--secondary-color);
                margin-bottom: 20px;
            }

            .nav-section {
                margin-bottom: 15px;
            }

            .nav-section-title {
                font-weight: 600;
                color: var(--text-color);
                padding: 8px 12px;
                cursor: pointer;
                border-radius: 4px;
                transition: background-color 0.2s;
            }

            .nav-section-title:hover {
                background-color: var(--bg-color);
            }

            .nav-items {
                padding-left: 20px;
                margin-top: 5px;
            }

            .nav-item {
                padding: 6px 12px;
                color: var(--secondary-color);
                cursor: pointer;
                border-radius: 4px;
                font-size: 0.9rem;
                transition: all 0.2s;
            }

            .nav-item:hover {
                background-color: var(--bg-color);
                color: var(--primary-color);
            }

            .nav-item.active {
                background-color: #e8f0fe;
                color: var(--primary-color);
            }

            /* Main Content */
            .main-content {
                flex: 1;
                margin-left: 280px;
                padding: 30px;
            }

            .section {
                margin-bottom: 40px;
            }

            .section-title {
                font-size: 1.8rem;
                color: var(--text-color);
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid var(--border-color);
            }

            .content-block {
                background-color: var(--card-bg);
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }

            .content-block h3 {
                font-size: 1.1rem;
                color: var(--secondary-color);
                margin-bottom: 15px;
                padding-bottom: 8px;
                border-bottom: 1px solid var(--border-color);
            }

            /* Tables */
            .table-container {
                overflow-x: auto;
            }

            .data-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9rem;
            }

            .data-table th {
                background-color: var(--bg-color);
                padding: 12px;
                text-align: left;
                font-weight: 600;
                border-bottom: 2px solid var(--border-color);
            }

            .data-table td {
                padding: 10px 12px;
                border-bottom: 1px solid var(--border-color);
            }

            .data-table tr:hover {
                background-color: var(--bg-color);
            }

            .kv-table {
                width: 100%;
            }

            .kv-table td {
                padding: 8px 12px;
                border-bottom: 1px solid var(--border-color);
            }

            .kv-table .key {
                font-weight: 600;
                width: 40%;
                color: var(--secondary-color);
            }

            .kv-table .value {
                font-family: 'Consolas', 'Monaco', monospace;
            }

            /* Figures */
            .figure-container {
                min-height: 400px;
            }

            /* Text */
            .text-content {
                font-size: 1rem;
                line-height: 1.8;
            }

            /* Responsive */
            @media (max-width: 768px) {
                .sidebar {
                    width: 100%;
                    height: auto;
                    position: relative;
                }

                .main-content {
                    margin-left: 0;
                }

                .container {
                    flex-direction: column;
                }
            }
        </style>
        """

    def _get_scripts(self) -> str:
        """Get JavaScript for interactivity."""
        return """
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Smooth scrolling for navigation
                document.querySelectorAll('.nav-item').forEach(item => {
                    item.addEventListener('click', function() {
                        const targetId = this.getAttribute('data-target');
                        const target = document.getElementById(targetId);
                        if (target) {
                            target.scrollIntoView({ behavior: 'smooth' });

                            // Update active state
                            document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                            this.classList.add('active');
                        }
                    });
                });

                // Section toggle
                document.querySelectorAll('.nav-section-title').forEach(title => {
                    title.addEventListener('click', function() {
                        const items = this.nextElementSibling;
                        if (items) {
                            items.style.display = items.style.display === 'none' ? 'block' : 'none';
                        }
                    });
                });
            });
        </script>
        """

    def generate_html(self) -> str:
        """Generate the complete HTML report."""
        # Build navigation
        nav_html = ""
        for section_name, contents in self.sections.items():
            section_id = section_name.lower().replace(" ", "-")
            nav_items = ""
            for content_name in contents.keys():
                content_id = f"{section_id}-{content_name.lower().replace(' ', '-')}"
                nav_items += f'<div class="nav-item" data-target="{content_id}">{content_name}</div>\n'

            nav_html += f"""
            <div class="nav-section">
                <div class="nav-section-title">{section_name}</div>
                <div class="nav-items">{nav_items}</div>
            </div>
            """

        # Build main content
        content_html = ""
        for section_name, contents in self.sections.items():
            section_id = section_name.lower().replace(" ", "-")
            section_content = ""

            for content_name, content in contents.items():
                content_id = f"{section_id}-{content_name.lower().replace(' ', '-')}"
                rendered = self._render_content(content_name, content)
                section_content += f'<div id="{content_id}">{rendered}</div>'

            content_html += f"""
            <div class="section" id="{section_id}">
                <h2 class="section-title">{section_name}</h2>
                {section_content}
            </div>
            """

        timestamp = self.created_at.strftime("%Y-%m-%d %H:%M:%S")

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.title}</title>
            <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
            {self._get_styles()}
        </head>
        <body>
            <div class="container">
                <nav class="sidebar">
                    <h1>{self.title}</h1>
                    <div class="timestamp">Generated: {timestamp}</div>
                    {nav_html}
                </nav>
                <main class="main-content">
                    {content_html}
                </main>
            </div>
            {self._get_scripts()}
        </body>
        </html>
        """

    def save(self, path: str | Path) -> None:
        """Save report to HTML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self.generate_html()
        path.write_text(html)
        print(f"Report saved to: {path}")


class QuantReportGenerator:
    """
    Generate comprehensive quant research reports.

    Creates visualizations and tables from pipeline results.
    """

    def __init__(self, title: str = "Quant Research Report"):
        self.builder = ReportBuilder(title)

    def add_price_chart(
        self,
        data: pd.DataFrame,
        symbol: str,
        indicators: dict[str, pd.Series] | None = None,
    ) -> go.Figure:
        """Create price chart with optional indicators."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} Price", "Volume"),
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="Price",
            ),
            row=1, col=1,
        )

        # Add indicators
        if indicators:
            colors = px.colors.qualitative.Set2
            for i, (name, series) in enumerate(indicators.items()):
                if "sma" in name.lower() or "ema" in name.lower():
                    fig.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=series,
                            name=name,
                            line=dict(color=colors[i % len(colors)], width=1),
                        ),
                        row=1, col=1,
                    )

        # Volume
        if "volume" in data.columns:
            colors = ["red" if c < o else "green" for c, o in zip(data["close"], data["open"])]
            fig.add_trace(
                go.Bar(x=data.index, y=data["volume"], name="Volume", marker_color=colors),
                row=2, col=1,
            )

        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        return fig

    def add_returns_distribution(self, returns: pd.Series, title: str = "Returns") -> go.Figure:
        """Create returns distribution histogram."""
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name="Returns",
            marker_color="#1a73e8",
            opacity=0.7,
        ))

        # Add normal distribution overlay
        mean, std = returns.mean(), returns.std()
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
        scale_factor = len(returns) * (returns.max() - returns.min()) / 50

        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist * scale_factor,
            name="Normal Distribution",
            line=dict(color="red", width=2),
        ))

        fig.update_layout(
            title=f"{title} Distribution",
            xaxis_title="Return",
            yaxis_title="Frequency",
            height=400,
        )

        return fig

    def add_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
        ))

        fig.update_layout(
            title="Correlation Matrix",
            height=500,
            width=600,
        )

        return fig

    def add_equity_curve(
        self,
        equity: pd.Series,
        benchmark: pd.Series | None = None,
        title: str = "Equity Curve",
    ) -> go.Figure:
        """Create equity curve chart."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity,
            name="Strategy",
            line=dict(color="#1a73e8", width=2),
        ))

        if benchmark is not None:
            fig.add_trace(go.Scatter(
                x=benchmark.index,
                y=benchmark,
                name="Benchmark",
                line=dict(color="#5f6368", width=1, dash="dash"),
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            height=400,
            hovermode="x unified",
        )

        return fig

    def add_drawdown_chart(self, equity: pd.Series) -> go.Figure:
        """Create drawdown chart."""
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#ea4335", width=1),
            fillcolor="rgba(234, 67, 53, 0.3)",
        ))

        fig.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=300,
        )

        return fig

    def add_feature_importance(
        self,
        importance: dict[str, float],
        top_n: int = 15,
    ) -> go.Figure:
        """Create feature importance bar chart."""
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, values = zip(*sorted_imp)

        fig = go.Figure(go.Bar(
            x=list(values),
            y=list(features),
            orientation="h",
            marker_color="#34a853",
        ))

        fig.update_layout(
            title=f"Top {top_n} Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400,
            yaxis=dict(autorange="reversed"),
        )

        return fig

    def add_efficient_frontier(
        self,
        returns: np.ndarray,
        volatilities: np.ndarray,
        optimal_point: tuple[float, float] | None = None,
    ) -> go.Figure:
        """Create efficient frontier plot."""
        fig = go.Figure()

        # Frontier line
        fig.add_trace(go.Scatter(
            x=volatilities * 100,
            y=returns * 100,
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="#1a73e8", width=2),
        ))

        # Optimal portfolio point
        if optimal_point:
            fig.add_trace(go.Scatter(
                x=[optimal_point[1] * 100],
                y=[optimal_point[0] * 100],
                mode="markers",
                name="Optimal Portfolio",
                marker=dict(color="#ea4335", size=15, symbol="star"),
            ))

        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Volatility (%)",
            yaxis_title="Expected Return (%)",
            height=500,
        )

        return fig

    def generate_from_pipeline(self, pipeline_result: Any) -> None:
        """Generate report from pipeline results."""
        # Data section
        self.builder.add_section("Data Overview")

        if hasattr(pipeline_result, "price_data") and pipeline_result.price_data:
            symbols = list(pipeline_result.price_data.keys())
            self.builder.add_content(
                "Data Overview",
                "Summary",
                {
                    "Symbols": ", ".join(symbols),
                    "Period": f"{pipeline_result.combined_prices.index[0].date()} to {pipeline_result.combined_prices.index[-1].date()}",
                    "Trading Days": len(pipeline_result.combined_prices),
                },
            )

            # Price charts for first 3 symbols
            for symbol in symbols[:3]:
                data = pipeline_result.price_data[symbol]
                indicators = pipeline_result.indicators_data.get(symbol, {})
                fig = self.add_price_chart(data, symbol, indicators)
                self.builder.add_content("Data Overview", f"{symbol} Chart", fig)

        # Portfolio section
        if hasattr(pipeline_result, "portfolio_stats") and pipeline_result.portfolio_stats is not None:
            self.builder.add_section("Portfolio Analysis")
            self.builder.add_content(
                "Portfolio Analysis",
                "Asset Statistics",
                pipeline_result.portfolio_stats,
            )

            if pipeline_result.portfolio_correlation is not None:
                fig = self.add_correlation_heatmap(pipeline_result.portfolio_correlation)
                self.builder.add_content("Portfolio Analysis", "Correlation", fig)

            if pipeline_result.optimized_portfolios:
                for name, data in pipeline_result.optimized_portfolios.items():
                    self.builder.add_content(
                        "Portfolio Analysis",
                        f"{name.replace('_', ' ').title()} Portfolio",
                        data["metrics"],
                    )

        # Backtest section
        if hasattr(pipeline_result, "backtest_results") and pipeline_result.backtest_results:
            self.builder.add_section("Backtest Results")

            # Summary table
            results_data = []
            for symbol, result in pipeline_result.backtest_results.items():
                results_data.append({
                    "Symbol": symbol,
                    "Return": f"{result.total_return:.2%}",
                    "Sharpe": f"{result.sharpe_ratio:.2f}",
                    "Max DD": f"{result.max_drawdown:.2%}",
                    "Trades": result.total_trades,
                })

            self.builder.add_content(
                "Backtest Results",
                "Summary",
                pd.DataFrame(results_data),
            )

    def save(self, path: str | Path) -> None:
        """Save report to file."""
        self.builder.save(path)


def generate_report(
    data_dict: dict,
    output_path: str | Path,
    title: str = "Quant Research Report",
) -> None:
    """
    Generate HTML report from nested dictionary structure.

    Args:
        data_dict: Nested dictionary with sections and content
        output_path: Output file path
        title: Report title
    """
    builder = ReportBuilder(title)

    for section_name, section_content in data_dict.items():
        builder.add_section(section_name)
        if isinstance(section_content, dict):
            for content_name, content in section_content.items():
                builder.add_content(section_name, content_name, content)
        else:
            builder.add_content(section_name, section_name, section_content)

    builder.save(output_path)
