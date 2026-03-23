"""KOSPI 일봉 5년치 가설 테스트 메인 스크립트

사용법:
    pip install -r requirements.txt
    python main.py
"""

from data_fetcher import fetch_kospi_daily, add_features
from hypothesis_tests import HypothesisTester


def main():
    # 1. 데이터 수집 (5년치 일봉)
    print("KOSPI 5년치 일봉 데이터를 가져오는 중...")
    df = fetch_kospi_daily(years=5)
    df = add_features(df)

    print(f"\n데이터 기본 통계:")
    print(f"  시작일: {df.index[0].date()}")
    print(f"  종료일: {df.index[-1].date()}")
    print(f"  거래일 수: {len(df)}")
    print(f"  일평균 수익률: {df['Daily_Return'].mean():.4%}")
    print(f"  일별 표준편차: {df['Daily_Return'].std():.4%}")
    print(f"  연환산 변동성: {df['Daily_Return'].std() * (252**0.5):.2%}")
    print(f"  최고종가: {df['Close'].max():,.0f}")
    print(f"  최저종가: {df['Close'].min():,.0f}")

    # 2. 가설 테스트 실행
    tester = HypothesisTester(df, alpha=0.05)
    results_df = tester.run_all()

    # 3. 결과 저장
    results_df.to_csv("hypothesis_test_results.csv", index=False, encoding="utf-8-sig")
    print("\n결과가 hypothesis_test_results.csv에 저장되었습니다.")

    # 4. 결과 요약 테이블 출력
    print("\n" + "=" * 80)
    print(f"{'테스트명':<30} {'p-value':>10} {'기각여부':>8}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        status = "** 기각 **" if row["H0 기각"] else "채택"
        print(f"{row['테스트명']:<30} {row['p-value']:>10.6f} {status:>8}")
    print("=" * 80)


if __name__ == "__main__":
    main()
