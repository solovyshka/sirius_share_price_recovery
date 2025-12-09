from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")


class StationarityTester:
    """
    Обертка над популярными тестами стационарности:
    1. ADF (Augmented Dickey-Fuller)
    2. KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
    3. PP (Phillips-Perron)
    4. Визуализация временного ряда
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results: Dict[str, Dict[str, Any]] = {}

    def adf_test(
        self,
        series: pd.Series,
        regression: str = "c",
        maxlag: Optional[int] = None,
        autolag: str = "AIC",
    ) -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller тест на единичные корни.

        Parameters
        ----------
        series: pd.Series
            Временной ряд для тестирования
        regression: str, default "c"
            Тип регрессии: 'c' - с константой, 'ct' - с константой и трендом,
            'ctt' - с константой, линейным и квадратичным трендом, 'n' - без константы
        maxlag: int, optional
            Максимальный лаг для автовыбора
        autolag: str, default "AIC"
            Метод выбора лага: 'AIC', 'BIC', 't-stat'
        """
        cleaned = series.dropna()
        result = adfuller(
            cleaned, regression=regression, maxlag=maxlag, autolag=autolag
        )

        test_result = {
            "test": "ADF",
            "test_statistic": result[0],
            "p_value": result[1],
            "critical_values": result[4],
            "n_lags": result[2],
            "n_obs": result[3],
            "stationary": result[1] < self.alpha,
            "h0": "Ряд имеет единичный корень (нестационарен)",
            "h1": "Ряд стационарен",
            "conclusion": "Стационарен" if result[1] < self.alpha else "Нестационарен",
        }

        self.results["adf"] = test_result
        return test_result

    def kpss_test(
        self, series: pd.Series, regression: str = "c", nlags: Optional[int] = "auto"
    ) -> Dict[str, Any]:
        """
        KPSS тест на стационарность.

        Parameters
        ----------
        series: pd.Series
            Ряд для тестирования
        regression: str, default "c"
            'c' - стационарность вокруг уровня, 'ct' - вокруг детерминированного тренда
        nlags: int or str, default "auto"
            Количество лагов для расчета long-run variance
        """
        cleaned = series.dropna()
        result = kpss(cleaned, regression=regression, nlags=nlags)

        test_result = {
            "test": "KPSS",
            "test_statistic": result[0],
            "p_value": result[1],
            "critical_values": result[3],
            "n_lags": result[2],
            "stationary": result[1] > self.alpha,
            "conclusion": "Стационарен" if result[1] > self.alpha else "Нестационарен",
        }

        self.results["kpss"] = test_result
        return test_result

    def pp_test(self, series: pd.Series, trend: str = "c") -> Dict[str, Any]:
        """
        Phillips-Perron тест на единичные корни.

        Parameters
        ----------
        series: pd.Series
            Временной ряд для тестирования
        trend: str, default "c"
            'c' - с константой, 'ct' - с константой и трендом
        """
        cleaned = series.dropna()
        result = PhillipsPerron(cleaned, trend=trend)

        test_result = {
            "test": "Phillips-Perron",
            "test_statistic": result.stat,
            "p_value": result.pvalue,
            "critical_values": result.critical_values,
            "n_obs": result.nobs,
            "stationary": result.pvalue < self.alpha,
            "conclusion": (
                "Стационарен" if result.pvalue < self.alpha else "Нестационарен"
            ),
        }

        self.results["pp"] = test_result
        return test_result

    def run_all_tests(self, series: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Запускает все тесты стационарности и печатает краткий вывод.
        """
        print(f"Длина ряда: {len(series)}")
        print(f"Уровень значимости: {self.alpha}")

        adf_result = self.adf_test(series)
        kpss_result = self.kpss_test(series)
        pp_result = self.pp_test(series)

        self._print_test_result(adf_result, series)
        self._print_test_result(kpss_result, series)
        self._print_test_result(pp_result, series)
        self.visualisation(series)

        if "adf" in self.results and "kpss" in self.results:
            adf_stationary = self.results["adf"]["stationary"]
            kpss_stationary = self.results["kpss"]["stationary"]

            if adf_stationary and kpss_stationary:
                print("Ряд стационарен")
            elif not adf_stationary and not kpss_stationary:
                print("Ряд нестационарен")
            elif adf_stationary and not kpss_stationary:
                print("Ряд, вероятно, стационарен вокруг детерминированного тренда")
                print("(ADF отвергает H0, KPSS не отвергает H0)")
            elif not adf_stationary and kpss_stationary:
                print("Противоречивые результаты. Требуется дополнительный анализ.")
                print("(ADF не отвергает H0, KPSS отвергает H0)")
        return self.results

    def _print_test_result(self, result: Dict[str, Any], series: pd.Series) -> None:
        """Печать результатов теста в читаемом формате."""
        print(f"{result['test']} Test:")
        print(f"  Тестовая статистика: {result['test_statistic']:.4f}")
        print(f"  P-value: {result['p_value']:.4f}")
        print(
            f"  Заключение: {result['conclusion']} (p-value {'<' if result['stationary'] else '>'} {self.alpha})"
        )

        if "critical_values" in result:
            print("  Критические значения:")
            for key, value in result["critical_values"].items():
                print(f"    {key}: {value:.4f}")

    def get_stationarity_summary(self) -> pd.DataFrame:
        """
        выход - pd.DataFrame с результатами всех тестов
        """
        summary_data = []
        for test_name, result in self.results.items():
            summary_data.append(
                {
                    "Test": result["test"],
                    "Statistic": f"{result['test_statistic']:.4f}",
                    "P-value": f"{result['p_value']:.4f}",
                    "Stationary": result["stationary"],
                    "Conclusion": result["conclusion"],
                }
            )

        return pd.DataFrame(summary_data)

    def visualisation(self, series: pd.Series) -> None:
        plt.plot(series)
        plt.title("Визуализация временного ряда")
        plt.xlabel("Дата")
        plt.ylabel("Значения")
        plt.show()


# example usage
# stat = StationarityTester(alpha=0.05)
# results1 = stat.run_all_tests(data['High'])
# print(tester1.get_stationarity_summary())
