"""KOSPI 일봉 데이터 기반 가설 테스트 모듈

테스트 목록:
1. 랜덤워크 검정 (수익률이 정규분포를 따르는가?)
2. 요일 효과 검정 (특정 요일에 수익률 차이가 있는가?)
3. 월별 효과 검정 (특정 월에 수익률 차이가 있는가? - Sell in May)
4. 평균 수익률 검정 (일평균 수익률이 0과 다른가?)
5. 변동성 클러스터링 검정 (ARCH 효과가 존재하는가?)
6. 자기상관 검정 (수익률에 자기상관이 있는가?)
7. 단위근 검정 (가격이 단위근을 갖는가? - 비정상성)
8. 골든크로스 효과 검정 (골든크로스 후 수익률이 유의한가?)
9. 변동성 비대칭 검정 (하락 시 변동성이 더 큰가? - Leverage Effect)
10. 추세 지속성 검정 (Hurst Exponent)
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from arch import arch_model


class HypothesisTester:
    """KOSPI 가설 테스트 클래스"""

    def __init__(self, df: pd.DataFrame, alpha: float = 0.05):
        self.df = df.copy()
        self.alpha = alpha
        self.results = []

    def _record(self, name: str, h0: str, statistic: float, p_value: float,
                conclusion: str):
        result = {
            "테스트명": name,
            "귀무가설(H0)": h0,
            "검정통계량": round(statistic, 4),
            "p-value": round(p_value, 6),
            "유의수준": self.alpha,
            "결론": conclusion,
            "H0 기각": p_value < self.alpha,
        }
        self.results.append(result)
        return result

    def _print_result(self, r: dict):
        emoji = "REJECT" if r["H0 기각"] else "FAIL TO REJECT"
        print(f"\n{'='*60}")
        print(f"[{r['테스트명']}]")
        print(f"  H0: {r['귀무가설(H0)']}")
        print(f"  검정통계량 = {r['검정통계량']}, p-value = {r['p-value']}")
        print(f"  결론: {r['결론']} ({emoji} H0 at α={self.alpha})")

    def test_normality(self) -> dict:
        """1. 정규성 검정 (Jarque-Bera)"""
        returns = self.df["Daily_Return"].dropna()
        stat, p = stats.jarque_bera(returns)

        if p < self.alpha:
            conclusion = "일별 수익률은 정규분포를 따르지 않습니다 (팻테일 존재)"
        else:
            conclusion = "일별 수익률이 정규분포를 따른다고 볼 수 있습니다"

        r = self._record("정규성 검정 (Jarque-Bera)",
                         "일별 수익률은 정규분포를 따른다", stat, p, conclusion)
        self._print_result(r)
        return r

    def test_weekday_effect(self) -> dict:
        """2. 요일 효과 검정 (Kruskal-Wallis)"""
        groups = [g["Daily_Return"].values
                  for _, g in self.df.groupby("Weekday")]
        stat, p = stats.kruskal(*groups)

        weekday_means = self.df.groupby("Weekday")["Daily_Return"].mean()
        best = ["월", "화", "수", "목", "금"][weekday_means.idxmax()]
        worst = ["월", "화", "수", "목", "금"][weekday_means.idxmin()]

        if p < self.alpha:
            conclusion = f"요일 효과가 존재합니다 (최고: {best}요일, 최저: {worst}요일)"
        else:
            conclusion = "요일에 따른 수익률 차이가 유의하지 않습니다"

        r = self._record("요일 효과 검정 (Kruskal-Wallis)",
                         "요일에 따른 수익률 차이가 없다", stat, p, conclusion)
        self._print_result(r)
        return r

    def test_monthly_effect(self) -> dict:
        """3. 월별 효과 검정 (Kruskal-Wallis) - Sell in May?"""
        groups = [g["Daily_Return"].values
                  for _, g in self.df.groupby("Month")]
        stat, p = stats.kruskal(*groups)

        monthly_means = self.df.groupby("Month")["Daily_Return"].mean()
        best_month = monthly_means.idxmax()
        worst_month = monthly_means.idxmin()

        if p < self.alpha:
            conclusion = (f"월별 효과가 존재합니다 "
                          f"(최고: {best_month}월, 최저: {worst_month}월)")
        else:
            conclusion = "월에 따른 수익률 차이가 유의하지 않습니다"

        r = self._record("월별 효과 검정 (Sell in May?)",
                         "월에 따른 수익률 차이가 없다", stat, p, conclusion)
        self._print_result(r)
        return r

    def test_mean_return(self) -> dict:
        """4. 평균 수익률 검정 (t-test)"""
        returns = self.df["Daily_Return"].dropna()
        stat, p = stats.ttest_1samp(returns, 0)

        mean_ret = returns.mean()
        annual_ret = mean_ret * 252

        if p < self.alpha:
            direction = "양(+)" if mean_ret > 0 else "음(-)"
            conclusion = (f"일평균 수익률({mean_ret:.4%})이 0과 유의하게 다릅니다 "
                          f"({direction}, 연환산 {annual_ret:.2%})")
        else:
            conclusion = f"일평균 수익률({mean_ret:.4%})이 0과 유의하게 다르지 않습니다"

        r = self._record("평균 수익률 검정 (t-test)",
                         "일평균 수익률 = 0", stat, p, conclusion)
        self._print_result(r)
        return r

    def test_arch_effect(self) -> dict:
        """5. ARCH 효과 검정 (변동성 클러스터링)"""
        returns = self.df["Daily_Return"].dropna() * 100  # 퍼센트 변환

        try:
            am = arch_model(returns, vol="GARCH", p=1, q=1, mean="Zero")
            res = am.fit(disp="off")

            # ARCH LM test
            from statsmodels.stats.diagnostic import het_arch
            stat, p, _, _ = het_arch(returns.values, nlags=5)

            if p < self.alpha:
                conclusion = "ARCH 효과가 존재합니다 (변동성 클러스터링 확인)"
            else:
                conclusion = "ARCH 효과가 유의하지 않습니다"

        except Exception as e:
            stat, p = float("nan"), float("nan")
            conclusion = f"ARCH 검정 실패: {e}"

        r = self._record("ARCH 효과 검정 (변동성 클러스터링)",
                         "ARCH 효과가 없다 (동분산)", stat, p, conclusion)
        self._print_result(r)
        return r

    def test_autocorrelation(self) -> dict:
        """6. 자기상관 검정 (Ljung-Box)"""
        returns = self.df["Daily_Return"].dropna()
        lb_result = acorr_ljungbox(returns, lags=[10], return_df=True)
        stat = lb_result["lb_stat"].values[0]
        p = lb_result["lb_pvalue"].values[0]

        if p < self.alpha:
            conclusion = "수익률에 유의한 자기상관이 존재합니다 (비효율적 시장 가능성)"
        else:
            conclusion = "수익률에 유의한 자기상관이 없습니다 (약형 효율적 시장 가설 지지)"

        r = self._record("자기상관 검정 (Ljung-Box, lag=10)",
                         "수익률에 자기상관이 없다", stat, p, conclusion)
        self._print_result(r)
        return r

    def test_unit_root(self) -> dict:
        """7. 단위근 검정 (ADF) - 가격의 비정상성"""
        price = self.df["Close"].dropna()
        adf_stat, p, _, _, critical, _ = adfuller(price, autolag="AIC")

        if p < self.alpha:
            conclusion = "가격 시계열이 정상적(stationary)입니다 (단위근 없음)"
        else:
            conclusion = ("가격 시계열이 비정상적(non-stationary)입니다 "
                          "(단위근 존재, 랜덤워크 특성)")

        r = self._record("단위근 검정 (ADF)",
                         "가격 시계열에 단위근이 존재한다 (비정상)", adf_stat, p,
                         conclusion)
        self._print_result(r)
        return r

    def test_golden_cross(self) -> dict:
        """8. 골든크로스 효과 검정"""
        gc_dates = self.df[self.df["Golden_Cross"] == 1].index
        if len(gc_dates) < 3:
            r = self._record("골든크로스 효과 검정", "골든크로스 후 수익률 = 0",
                             float("nan"), float("nan"),
                             f"골든크로스 발생 횟수 부족 ({len(gc_dates)}회)")
            self._print_result(r)
            return r

        # 골든크로스 후 5일 수익률
        returns_after_gc = []
        for date in gc_dates:
            loc = self.df.index.get_loc(date)
            if loc + 5 < len(self.df):
                ret_5d = (self.df["Close"].iloc[loc + 5] /
                          self.df["Close"].iloc[loc] - 1)
                returns_after_gc.append(ret_5d)

        if len(returns_after_gc) < 3:
            r = self._record("골든크로스 효과 검정", "골든크로스 후 5일 수익률 = 0",
                             float("nan"), float("nan"), "샘플 부족")
            self._print_result(r)
            return r

        stat, p = stats.ttest_1samp(returns_after_gc, 0)
        mean_ret = np.mean(returns_after_gc)

        if p < self.alpha:
            conclusion = (f"골든크로스 후 5일 평균 수익률({mean_ret:.2%})이 "
                          f"유의합니다 (발생 {len(returns_after_gc)}회)")
        else:
            conclusion = (f"골든크로스 후 5일 수익률({mean_ret:.2%})은 "
                          f"유의하지 않습니다 (발생 {len(returns_after_gc)}회)")

        r = self._record("골든크로스 효과 검정 (5일 후)",
                         "골든크로스 후 5일 수익률 = 0", stat, p, conclusion)
        self._print_result(r)
        return r

    def test_leverage_effect(self) -> dict:
        """9. 레버리지 효과 검정 (하락 시 변동성 > 상승 시 변동성)"""
        returns = self.df["Daily_Return"].dropna()

        up_vol = returns[returns > 0].std()
        down_vol = returns[returns < 0].std()

        up_returns = returns[returns > 0].values
        down_returns = np.abs(returns[returns < 0].values)

        stat, p = stats.mannwhitneyu(down_returns, up_returns,
                                     alternative="greater")

        if p < self.alpha:
            conclusion = (f"레버리지 효과가 존재합니다 "
                          f"(하락변동성 {down_vol:.4f} > 상승변동성 {up_vol:.4f})")
        else:
            conclusion = (f"레버리지 효과가 유의하지 않습니다 "
                          f"(하락 {down_vol:.4f} vs 상승 {up_vol:.4f})")

        r = self._record("레버리지 효과 검정",
                         "하락 시 변동성 = 상승 시 변동성", stat, p, conclusion)
        self._print_result(r)
        return r

    def test_hurst_exponent(self) -> dict:
        """10. Hurst Exponent (추세 지속성 검정)
        H < 0.5: 평균회귀, H = 0.5: 랜덤워크, H > 0.5: 추세 지속
        """
        prices = self.df["Close"].dropna().values
        n = len(prices)

        max_lag = min(n // 2, 500)
        lags = range(2, max_lag)
        tau = []
        for lag in lags:
            pp = np.subtract(prices[lag:], prices[:-lag])
            tau.append(np.sqrt(np.std(pp)))

        log_lags = np.log(list(lags))
        log_tau = np.log(tau)

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_lags, log_tau)
        hurst = slope * 2

        if hurst < 0.45:
            conclusion = f"평균회귀 특성 (H={hurst:.3f} < 0.5)"
        elif hurst > 0.55:
            conclusion = f"추세 지속 특성 (H={hurst:.3f} > 0.5)"
        else:
            conclusion = f"랜덤워크에 가까움 (H={hurst:.3f} ≈ 0.5)"

        # H=0.5와의 차이를 t-test로 근사
        t_stat = (hurst - 0.5) / std_err if std_err > 0 else 0
        p_approx = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(lags) - 2))

        r = self._record("Hurst Exponent (추세 지속성)",
                         "H = 0.5 (랜덤워크)", round(hurst, 4), p_approx,
                         conclusion)
        self._print_result(r)
        return r

    def run_all(self) -> pd.DataFrame:
        """모든 가설 테스트를 실행하고 결과를 데이터프레임으로 반환합니다."""
        print("\n" + "=" * 60)
        print("  KOSPI 일봉 데이터 가설 테스트 결과")
        print("=" * 60)

        self.test_normality()
        self.test_weekday_effect()
        self.test_monthly_effect()
        self.test_mean_return()
        self.test_arch_effect()
        self.test_autocorrelation()
        self.test_unit_root()
        self.test_golden_cross()
        self.test_leverage_effect()
        self.test_hurst_exponent()

        print("\n" + "=" * 60)
        print("  요약")
        print("=" * 60)

        results_df = pd.DataFrame(self.results)
        rejected = results_df["H0 기각"].sum()
        total = len(results_df)
        print(f"\n총 {total}개 가설 중 {rejected}개 기각 (유의수준 α={self.alpha})")

        return results_df
