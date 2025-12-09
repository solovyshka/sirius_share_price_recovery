from __future__ import annotations

import pandas as pd

from Modeling.decompositions import holt_winters_decomposition, stl_decomposition
from Modeling.exog_forecasters import (
    EnsembleExogForecaster,
    LastValueExogForecaster,
    PerColumnAutoArimaExogForecaster,
    PerColumnExpSmoothingExogForecaster,
    PerColumnLastSlopeExogForecaster,
    PerColumnLinearRegExogForecaster,
    SeasonalNaiveExogForecaster,
    VarExogForecaster,
)
from Modeling.features_builders import (
    fracdiff_feature,
    rolling_features_feature,
    technical_indicator_feature,
)
from Modeling.forecast_pipeline import ForecastPipeline
from Modeling.residual_models import ArimaResidualModel, AutoArimaResidualModel
from Modeling.trend_forecasting import (
    trend_forecast_exponential,
    trend_forecast_last_slope,
    trend_forecast_linear_reg,
    trend_forecast_logistic,
)


def run_demo(
    name: str,
    pipeline: ForecastPipeline,
    raw_df: pd.DataFrame,
    steps: int,
    plot_tail: int,
):
    print(f"\n=== {name} (forecast) ===")
    result = pipeline.forecast(raw_df, steps=steps, plot=True, plot_last=plot_tail)
    print("Forecast tail:")
    print(result.forecast.tail())
    print("Stationarity summary:")
    print(result.stationarity_tests.get("summary"))
    return result


def evaluate_demo(
    name: str,
    pipeline: ForecastPipeline,
    raw_df: pd.DataFrame,
    test_size: int,
    plot_tail: int,
):
    print(f"\n=== {name} (evaluate) ===")
    eval_result = pipeline.evaluate(
        raw_df, test_size=test_size, plot=True, plot_last=plot_tail
    )
    print("Metrics:", eval_result["metrics"])
    print("Stationarity summary:")
    print(eval_result["stationarity_tests"].get("summary"))
    return eval_result


def run_all_demos():
    raw_df = pd.read_csv("DataBase/CLOSE.csv")

    steps = 30
    test_size = 200
    plot_tail = 400
    trend_window = 200

    rolling_feat = rolling_features_feature(window=5)
    fracdiff_feat = fracdiff_feature(
        diff_amt=0.3, log_smooth=True, column_name="fracdiff_0_3"
    )
    tech_feature = technical_indicator_feature(
        ticker="SBER", indicator_columns=["SMA_20", "EMA_12", "MACD", "RSI", "OBV"]
    )

    arima_factory = lambda: ArimaResidualModel(order=(1, 1, 1))
    auto_arima_factory = lambda: AutoArimaResidualModel(
        seasonal=False, max_p=4, max_q=4, max_d=2
    )

    exog_autoarima_factory = lambda: PerColumnAutoArimaExogForecaster(
        seasonal=False, max_p=2, max_q=2, max_d=1
    )
    last_value_exog_factory = lambda: LastValueExogForecaster()
    seasonal_naive_exog_factory = lambda: SeasonalNaiveExogForecaster(period=20)
    last_slope_exog_factory = lambda: PerColumnLastSlopeExogForecaster(window=8)
    linear_reg_exog_factory = lambda: PerColumnLinearRegExogForecaster()
    var_exog_factory = lambda: VarExogForecaster(maxlags=5, ic="aic")
    exp_smoothing_exog_factory = lambda: PerColumnExpSmoothingExogForecaster(
        smoothing_level=None, optimized=True
    )
    ensemble_exog_factory = lambda: EnsembleExogForecaster(
        forecaster_factories=[
            last_value_exog_factory,
            exp_smoothing_exog_factory,
            last_slope_exog_factory,
        ]
    )

    run_demo(
        "STL + ARIMA + logistic trend with rolling + fracdiff features",
        ForecastPipeline(
            ticker="SBER",
            feature_builders=[rolling_feat, fracdiff_feat],
            decomposition_fn=lambda s: stl_decomposition(
                s,
                trend_forecaster=trend_forecast_logistic(trend_window),
            ),
            model_factory=arima_factory,
        ),
        raw_df,
        steps,
        plot_tail,
    )

    evaluate_demo(
        "STL + ARIMA, last-slope trend, rolling window features",
        ForecastPipeline(
            ticker="SBER",
            feature_builders=[rolling_feat],
            decomposition_fn=lambda s: stl_decomposition(
                s,
                trend_forecaster=trend_forecast_last_slope(trend_window),
            ),
            model_factory=arima_factory,
        ),
        raw_df,
        test_size,
        plot_tail,
    )

    evaluate_demo(
        "STL + ARIMA, exponential trend, fracdiff exogenous feature",
        ForecastPipeline(
            ticker="SBER",
            feature_builders=[fracdiff_feat],
            decomposition_fn=lambda s: stl_decomposition(
                s,
                trend_forecaster=trend_forecast_exponential(trend_window),
            ),
            model_factory=arima_factory,
        ),
        raw_df,
        test_size,
        plot_tail,
    )

    evaluate_demo(
        "Holt-Winters + ARIMA, linear regression trend, technical indicators",
        ForecastPipeline(
            ticker="SBER",
            feature_builders=[tech_feature],
            decomposition_fn=lambda s: holt_winters_decomposition(
                s,
                trend_mode="add",
                seasonal_mode="add",
                trend_forecaster=trend_forecast_linear_reg(trend_window),
            ),
            model_factory=arima_factory,
        ),
        raw_df,
        test_size,
        plot_tail,
    )

    evaluate_demo(
        "STL + AutoARIMA residuals, logistic trend, rolling features",
        ForecastPipeline(
            ticker="SBER",
            feature_builders=[rolling_feat],
            decomposition_fn=lambda s: stl_decomposition(
                s,
                trend_forecaster=trend_forecast_logistic(trend_window),
            ),
            model_factory=auto_arima_factory,
        ),
        raw_df,
        test_size,
        plot_tail,
    )

    # too long to run locally
    # evaluate_demo(
    #     "STL + ARIMA (resid) + AutoARIMA exog forecast, rolling features",
    #     ForecastPipeline(
    #         ticker="SBER",
    #         feature_builders=[rolling_feat],
    #         decomposition_fn=lambda s: stl_decomposition(
    #         ),
    #         model_factory=arima_factory,
    #         exog_forecast_factory=exog_autoarima_factory,
    #     ),
    #     raw_df,
    #     test_size,
    #     plot_tail,
    # )

    evaluate_demo(
        "STL + ARIMA (resid) + LastValue exog forecast, rolling features",
        ForecastPipeline(
            ticker="SBER",
            feature_builders=[rolling_feat],
            decomposition_fn=lambda s: stl_decomposition(
                s,
                trend_forecaster=trend_forecast_logistic(trend_window),
            ),
            model_factory=arima_factory,
            exog_forecast_factory=last_value_exog_factory,
        ),
        raw_df,
        test_size,
        plot_tail,
    )

    evaluate_demo(
        "STL + ARIMA (resid) + SeasonalNaive exog forecast, rolling features",
        ForecastPipeline(
            ticker="SBER",
            feature_builders=[rolling_feat],
            decomposition_fn=lambda s: stl_decomposition(
                s,
                trend_forecaster=trend_forecast_logistic(trend_window),
            ),
            model_factory=arima_factory,
            exog_forecast_factory=seasonal_naive_exog_factory,
        ),
        raw_df,
        test_size,
        plot_tail,
    )

    evaluate_demo(
        "STL + ARIMA (resid) + LastSlope exog forecast, rolling features",
        ForecastPipeline(
            ticker="SBER",
            feature_builders=[rolling_feat],
            decomposition_fn=lambda s: stl_decomposition(
                s,
                trend_forecaster=trend_forecast_logistic(trend_window),
            ),
            model_factory=arima_factory,
            exog_forecast_factory=last_slope_exog_factory,
        ),
        raw_df,
        test_size,
        plot_tail,
    )

    evaluate_demo(
        "STL + ARIMA (resid) + LinearReg exog forecast, rolling features",
        ForecastPipeline(
            ticker="SBER",
            feature_builders=[rolling_feat],
            decomposition_fn=lambda s: stl_decomposition(
                s,
                trend_forecaster=trend_forecast_logistic(trend_window),
            ),
            model_factory=arima_factory,
            exog_forecast_factory=linear_reg_exog_factory,
        ),
        raw_df,
        test_size,
        plot_tail,
    )

    evaluate_demo(
        "STL + ARIMA (resid) + VAR exog forecast, rolling features",
        ForecastPipeline(
            ticker="SBER",
            feature_builders=[rolling_feat],
            decomposition_fn=lambda s: stl_decomposition(
                s,
                trend_forecaster=trend_forecast_logistic(trend_window),
            ),
            model_factory=arima_factory,
            exog_forecast_factory=var_exog_factory,
        ),
        raw_df,
        test_size,
        plot_tail,
    )

    evaluate_demo(
        "STL + ARIMA (resid) + ExpSmoothing exog forecast, rolling features",
        ForecastPipeline(
            ticker="SBER",
            feature_builders=[rolling_feat],
            decomposition_fn=lambda s: stl_decomposition(
                s,
                trend_forecaster=trend_forecast_logistic(trend_window),
            ),
            model_factory=arima_factory,
            exog_forecast_factory=exp_smoothing_exog_factory,
        ),
        raw_df,
        test_size,
        plot_tail,
    )

    evaluate_demo(
        "STL + ARIMA (resid) + Ensemble exog forecast, rolling features",
        ForecastPipeline(
            ticker="SBER",
            feature_builders=[rolling_feat],
            decomposition_fn=lambda s: stl_decomposition(
                s,
                trend_forecaster=trend_forecast_logistic(trend_window),
            ),
            model_factory=arima_factory,
            exog_forecast_factory=ensemble_exog_factory,
        ),
        raw_df,
        test_size,
        plot_tail,
    )


if __name__ == "__main__":
    run_all_demos()
