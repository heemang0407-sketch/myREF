"""KOSPI 일봉 데이터 수집 모듈 (5년치)"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_kospi_daily(years: int = 5) -> pd.DataFrame:
    """KOSPI 지수 일봉 데이터를 Yahoo Finance에서 가져옵니다.

    Args:
        years: 가져올 연도 수 (기본값 5년)

    Returns:
        일봉 OHLCV 데이터프레임
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    # ^KS11 = KOSPI Composite Index
    ticker = yf.Ticker("^KS11")
    df = ticker.history(start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        interval="1d")

    if df.empty:
        raise ValueError("KOSPI 데이터를 가져올 수 없습니다. 네트워크 연결을 확인하세요.")

    # 수익률 계산
    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = pd.Series(df["Close"]).apply(lambda x: x).pipe(
        lambda s: s.apply(lambda x: float("nan") if x <= 0 else x)
    )
    # 간단하게 로그 수익률 계산
    import numpy as np
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df.dropna(subset=["Daily_Return"], inplace=True)

    print(f"[데이터 수집 완료] 기간: {df.index[0].date()} ~ {df.index[-1].date()}, "
          f"총 {len(df)}개 거래일")

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """분석에 필요한 파생 변수를 추가합니다."""
    import numpy as np

    # 이동평균
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    # 변동성 (20일 rolling std)
    df["Volatility_20d"] = df["Daily_Return"].rolling(20).std() * np.sqrt(252)

    # 요일 (0=월, 4=금)
    df["Weekday"] = df.index.weekday

    # 월
    df["Month"] = df.index.month

    # 골든크로스 / 데드크로스 시그널
    df["Golden_Cross"] = ((df["MA5"] > df["MA20"]) &
                          (df["MA5"].shift(1) <= df["MA20"].shift(1))).astype(int)
    df["Dead_Cross"] = ((df["MA5"] < df["MA20"]) &
                        (df["MA5"].shift(1) >= df["MA20"].shift(1))).astype(int)

    return df


if __name__ == "__main__":
    df = fetch_kospi_daily(5)
    df = add_features(df)
    print(df.tail())
    print(f"\n컬럼: {list(df.columns)}")
