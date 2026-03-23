"""코스피 선물 시그널 기반 백테스트

전략:
  - 전일 코스피200 선물(프록시: ^KS200)이 상승 마감 → 다음날 시가 매수, 당일 종가 매도
  - 전일 코스피200 선물이 하락 마감 → 매수 미진행 (현금 보유)
  - 초기 투자금: 1,000만원
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def fetch_data(years: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """KOSPI 지수와 KOSPI200(선물 프록시) 데이터를 가져옵니다."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # KOSPI 종합지수 (매매 대상)
    kospi = yf.Ticker("^KS11").history(start=start_str, end=end_str, interval="1d")

    # KOSPI 200 (선물 방향 시그널용 프록시)
    # Yahoo Finance에서 코스피200 선물 연속계약이 불안정하므로 ^KS200 현물 사용
    kospi200 = yf.Ticker("^KS200").history(start=start_str, end=end_str, interval="1d")

    if kospi.empty or kospi200.empty:
        raise ValueError("데이터를 가져올 수 없습니다. 네트워크를 확인하세요.")

    # 인덱스를 날짜만으로 통일 (timezone 제거)
    kospi.index = kospi.index.tz_localize(None).normalize()
    kospi200.index = kospi200.index.tz_localize(None).normalize()

    print(f"[KOSPI]    {kospi.index[0].date()} ~ {kospi.index[-1].date()}, {len(kospi)}일")
    print(f"[KOSPI200] {kospi200.index[0].date()} ~ {kospi200.index[-1].date()}, {len(kospi200)}일")

    return kospi, kospi200


def run_backtest(initial_capital: float = 10_000_000, years: int = 5) -> pd.DataFrame:
    """백테스트 실행

    Args:
        initial_capital: 초기 투자금 (기본 1,000만원)
        years: 백테스트 기간 (기본 5년)

    Returns:
        일별 백테스트 결과 데이터프레임
    """
    kospi, kospi200 = fetch_data(years)

    # KOSPI200 전일 수익률 (선물 방향 시그널)
    kospi200["Prev_Return"] = kospi200["Close"].pct_change()
    # 전일 상승 여부 → 당일 시그널
    kospi200["Signal"] = (kospi200["Prev_Return"] > 0).astype(int).shift(1)

    # 두 데이터를 날짜 기준으로 병합
    merged = kospi[["Open", "Close"]].copy()
    merged.columns = ["KOSPI_Open", "KOSPI_Close"]
    merged["Signal"] = kospi200["Signal"]
    merged.dropna(subset=["Signal"], inplace=True)

    # 일별 시가매수 → 종가매도 수익률
    merged["Intraday_Return"] = (merged["KOSPI_Close"] / merged["KOSPI_Open"]) - 1

    # 전략 수익률: 시그널=1이면 매매, 0이면 현금(수익률 0)
    merged["Strategy_Return"] = merged["Signal"] * merged["Intraday_Return"]

    # 누적 자산
    merged["Buy_Hold_Equity"] = initial_capital * (1 + merged["Intraday_Return"]).cumprod()
    merged["Strategy_Equity"] = initial_capital * (1 + merged["Strategy_Return"]).cumprod()

    # 통계 계산
    total_days = len(merged)
    trade_days = int(merged["Signal"].sum())
    no_trade_days = total_days - trade_days

    strategy_total_return = merged["Strategy_Equity"].iloc[-1] / initial_capital - 1
    buyhold_total_return = merged["Buy_Hold_Equity"].iloc[-1] / initial_capital - 1

    # 연환산 수익률
    n_years = (merged.index[-1] - merged.index[0]).days / 365.25
    strategy_cagr = (1 + strategy_total_return) ** (1 / n_years) - 1
    buyhold_cagr = (1 + buyhold_total_return) ** (1 / n_years) - 1

    # 최대 낙폭 (MDD)
    strategy_peak = merged["Strategy_Equity"].cummax()
    strategy_dd = (merged["Strategy_Equity"] - strategy_peak) / strategy_peak
    strategy_mdd = strategy_dd.min()

    buyhold_peak = merged["Buy_Hold_Equity"].cummax()
    buyhold_dd = (merged["Buy_Hold_Equity"] - buyhold_peak) / buyhold_peak
    buyhold_mdd = buyhold_dd.min()

    # 샤프 비율 (무위험 수익률 3.5% 가정)
    rf_daily = 0.035 / 252
    strategy_sharpe = ((merged["Strategy_Return"].mean() - rf_daily) /
                       merged["Strategy_Return"].std() * np.sqrt(252))
    buyhold_sharpe = ((merged["Intraday_Return"].mean() - rf_daily) /
                      merged["Intraday_Return"].std() * np.sqrt(252))

    # 승률 (매매일 기준)
    trade_returns = merged.loc[merged["Signal"] == 1, "Intraday_Return"]
    win_rate = (trade_returns > 0).mean()
    avg_win = trade_returns[trade_returns > 0].mean() if (trade_returns > 0).any() else 0
    avg_loss = trade_returns[trade_returns <= 0].mean() if (trade_returns <= 0).any() else 0

    # 결과 출력
    print("\n" + "=" * 65)
    print("  코스피 선물 시그널 백테스트 결과")
    print("  전략: 선물 전일 상승 → 시가매수/종가매도, 하락 → 미진행")
    print("=" * 65)

    print(f"\n  기간: {merged.index[0].date()} ~ {merged.index[-1].date()} ({n_years:.1f}년)")
    print(f"  초기 투자금: {initial_capital:>15,.0f}원")

    print(f"\n{'':>30} {'전략':>15} {'매일매매(B&H)':>15}")
    print(f"  {'-'*60}")
    print(f"  {'최종 자산':>28} {merged['Strategy_Equity'].iloc[-1]:>14,.0f}원 "
          f"{merged['Buy_Hold_Equity'].iloc[-1]:>14,.0f}원")
    print(f"  {'총 수익률':>28} {strategy_total_return:>14.2%} "
          f"{buyhold_total_return:>14.2%}")
    print(f"  {'연환산 수익률(CAGR)':>28} {strategy_cagr:>14.2%} "
          f"{buyhold_cagr:>14.2%}")
    print(f"  {'최대낙폭(MDD)':>28} {strategy_mdd:>14.2%} "
          f"{buyhold_mdd:>14.2%}")
    print(f"  {'샤프비율':>28} {strategy_sharpe:>14.2f} "
          f"{buyhold_sharpe:>14.2f}")

    print(f"\n  [매매 상세]")
    print(f"  {'총 거래일':>28} {total_days:>10}일")
    print(f"  {'매매 진행일':>28} {trade_days:>10}일 ({trade_days/total_days:.1%})")
    print(f"  {'매매 미진행일':>28} {no_trade_days:>10}일")
    print(f"  {'승률':>28} {win_rate:>10.1%}")
    print(f"  {'평균 수익 (이긴 날)':>28} {avg_win:>10.3%}")
    print(f"  {'평균 손실 (진 날)':>28} {avg_loss:>10.3%}")
    print(f"  {'손익비':>28} {abs(avg_win/avg_loss) if avg_loss != 0 else 0:>10.2f}")
    print("=" * 65)

    # 연도별 수익률
    merged["Year"] = merged.index.year
    print(f"\n  [연도별 수익률]")
    print(f"  {'연도':>8} {'전략':>12} {'매일매매':>12} {'매매일수':>10}")
    print(f"  {'-'*45}")
    for year, group in merged.groupby("Year"):
        yr_strat = (1 + group["Strategy_Return"]).prod() - 1
        yr_bh = (1 + group["Intraday_Return"]).prod() - 1
        yr_trades = int(group["Signal"].sum())
        print(f"  {year:>8} {yr_strat:>12.2%} {yr_bh:>12.2%} {yr_trades:>10}")

    return merged


if __name__ == "__main__":
    results = run_backtest(initial_capital=10_000_000, years=5)
    results.to_csv("backtest_results.csv", encoding="utf-8-sig")
    print("\n상세 결과가 backtest_results.csv에 저장되었습니다.")
