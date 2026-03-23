"""Sample usage of the IDX Python Wrapper."""

from __future__ import annotations

from idx_wrapper import IDXClient


def main() -> None:
    client = IDXClient()

    # ------------------------------------------------------------------
    # 1. Fetch the latest trading summary for a single stock
    # ------------------------------------------------------------------
    print("=" * 60)
    print("1. Stock price / trading summary for BBCA")
    print("=" * 60)
    bbca = client.get_stock_price("BBCA")
    if bbca:
        print(bbca)
    else:
        print("BBCA not found.")

    # ------------------------------------------------------------------
    # 2. Fetch the first 5 stocks from the full summary list
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. First 5 stocks from trading summary")
    print("=" * 60)
    summary = client.get_stock_summary(length=5)
    for stock in summary.get("data", []):
        print(stock)

    # ------------------------------------------------------------------
    # 3. Search for bank stocks by name
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. Stocks matching name 'bank' (first 5)")
    print("=" * 60)
    banks = client.get_stock_summary(name="bank", length=5)
    for stock in banks.get("data", []):
        print(stock)

    # ------------------------------------------------------------------
    # 4. Fetch company profile for a specific stock
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. Company profile for TLKM")
    print("=" * 60)
    tlkm_profile = client.get_company_profiles(code="TLKM", length=1)
    for company in tlkm_profile.get("data", []):
        print(company)

    # ------------------------------------------------------------------
    # 5. Fetch daily price data for BBCA in January 2024
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. Daily prices for BBCA — Jan 2024")
    print("=" * 60)
    daily = client.get_stocks_daily(
        code="BBCA",
        date_from="2024-01-01",
        date_to="2024-01-31",
        length=10,
    )
    for row in daily.get("data", []):
        print(row)

    # ------------------------------------------------------------------
    # 6. Fetch IHSG (Jakarta Composite) index statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("6. IHSG (COMPOSITE) index — last 5 monthly data points")
    print("=" * 60)
    ihsg = client.get_index_statistics(
        index_id="COMPOSITE", period="monthly", length=5
    )
    for row in ihsg.get("data", []):
        print(row)


if __name__ == "__main__":
    main()
