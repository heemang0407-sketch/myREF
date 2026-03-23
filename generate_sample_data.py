"""실제 KOSPI 통계 특성을 반영한 샘플 데이터 생성

Yahoo Finance 접속 불가 환경에서 백테스트 실행용.
실제 KOSPI 5년(2021~2025) 통계를 기반으로 현실적인 데이터를 생성합니다.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_kospi_sample(years: int = 5, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """실제 KOSPI 통계 특성을 반영한 샘플 데이터 생성

    실제 KOSPI 통계 기반 파라미터:
    - 일평균 수익률: ~0.02%
    - 일별 변동성: ~1.1%
    - 약간의 음의 왜도 (하락 시 더 큰 폭)
    - 팻테일 (첨도 > 3)
    - KOSPI200과 KOSPI의 상관계수: ~0.98
    """
    rng = np.random.default_rng(seed)

    # 거래일 생성 (주말 제외)
    start = datetime(2021, 3, 22)
    dates = []
    current = start
    total_trading_days = int(years * 252)
    while len(dates) < total_trading_days:
        if current.weekday() < 5:  # 월~금
            # 공휴일 대략 반영 (연 15일)
            if not (rng.random() < 15 / 252):
                dates.append(current)
        current += timedelta(days=1)

    n = len(dates)

    # KOSPI 수익률 생성 (t-분포로 팻테일 반영)
    mu_daily = 0.0002       # 일평균 수익률 0.02%
    sigma_daily = 0.011     # 일별 변동성 1.1%
    df_t = 5                # 자유도 5의 t-분포 (팻테일)

    # t-분포 기반 수익률
    kospi_returns = mu_daily + sigma_daily * rng.standard_t(df_t, size=n) / np.sqrt(df_t / (df_t - 2))

    # 변동성 클러스터링 반영 (GARCH-like)
    vol = np.ones(n) * sigma_daily
    for i in range(1, n):
        vol[i] = 0.9 * vol[i - 1] + 0.1 * abs(kospi_returns[i - 1])
        kospi_returns[i] = mu_daily + vol[i] * (kospi_returns[i] - mu_daily) / sigma_daily

    # KOSPI 가격 생성 (시작: 3,000)
    kospi_close = 3000 * np.exp(np.cumsum(kospi_returns))

    # 시가 = 전일 종가 기반 + 갭 노이즈
    gap_noise = rng.normal(0, 0.003, size=n)  # 갭 변동 0.3%
    kospi_open = np.roll(kospi_close, 1) * (1 + gap_noise)
    kospi_open[0] = 3000

    # 고가/저가
    intraday_range = np.abs(rng.normal(0.008, 0.004, size=n))
    kospi_high = np.maximum(kospi_open, kospi_close) * (1 + intraday_range / 2)
    kospi_low = np.minimum(kospi_open, kospi_close) * (1 - intraday_range / 2)

    # 거래량
    kospi_volume = rng.integers(300_000_000, 800_000_000, size=n)

    kospi_df = pd.DataFrame({
        "Open": kospi_open,
        "High": kospi_high,
        "Low": kospi_low,
        "Close": kospi_close,
        "Volume": kospi_volume,
    }, index=pd.DatetimeIndex(dates, name="Date"))

    # KOSPI200 생성 (KOSPI와 상관계수 ~0.98)
    corr_noise = rng.normal(0, 0.002, size=n)  # 약간의 차이
    k200_returns = kospi_returns * 1.02 + corr_noise  # 베타 약 1.02
    k200_close = 400 * np.exp(np.cumsum(k200_returns))

    k200_open = np.roll(k200_close, 1) * (1 + rng.normal(0, 0.003, size=n))
    k200_open[0] = 400
    k200_high = np.maximum(k200_open, k200_close) * (1 + intraday_range / 2)
    k200_low = np.minimum(k200_open, k200_close) * (1 - intraday_range / 2)

    kospi200_df = pd.DataFrame({
        "Open": k200_open,
        "High": k200_high,
        "Low": k200_low,
        "Close": k200_close,
        "Volume": rng.integers(50_000_000, 200_000_000, size=n),
    }, index=pd.DatetimeIndex(dates, name="Date"))

    return kospi_df, kospi200_df


if __name__ == "__main__":
    kospi, k200 = generate_kospi_sample()
    print(f"KOSPI: {len(kospi)}일, {kospi.index[0].date()} ~ {kospi.index[-1].date()}")
    print(f"  시작 {kospi['Close'].iloc[0]:.0f} → 종료 {kospi['Close'].iloc[-1]:.0f}")
    print(f"KOSPI200: {len(k200)}일")
    print(f"  시작 {k200['Close'].iloc[0]:.0f} → 종료 {k200['Close'].iloc[-1]:.0f}")
