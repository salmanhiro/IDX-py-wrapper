# IDX-py-wrapper

A Python wrapper for the public [Indonesia Stock Exchange (IDX)](https://www.idx.co.id/en/products/idx-data-services/) data APIs, exposed as a self-hosted REST service that can run locally or inside a Docker container.

---

## Features

| Endpoint | Description |
|---|---|
| `GET /stocks` | Trading summary for all (or filtered) listed stocks |
| `GET /stocks/{code}` | Latest trading summary for a single stock ticker |
| `GET /stocks/{code}/daily` | Daily OHLCV price history for a stock |
| `GET /companies` | Company profiles for listed IDX companies |
| `GET /index/{index_id}` | Historical statistics for a market index (e.g. IHSG) |
| `GET /brokers` | Trading summary per broker member |

Interactive API documentation is available at **`/docs`** (Swagger UI) once the server is running.

---

## Quick start

### Run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/docs> in your browser.

### Run with Docker

```bash
# Build and start the container
docker compose up --build
```

The API will be available at <http://localhost:8000>.

---

## Example requests

```bash
# Health check
curl http://localhost:8000/

# Latest price for BCA (BBCA)
curl http://localhost:8000/stocks/BBCA

# First 5 bank stocks
curl "http://localhost:8000/stocks?name=bank&length=5"

# Daily OHLCV for BBCA in January 2024
curl "http://localhost:8000/stocks/BBCA/daily?dateFrom=2024-01-01&dateTo=2024-01-31"

# IHSG (Composite) index – last 20 monthly data points
curl "http://localhost:8000/index/COMPOSITE?period=monthly"

# Company profile for Telkom Indonesia
curl "http://localhost:8000/companies?code=TLKM"

# Top 10 brokers
curl "http://localhost:8000/brokers?length=10"
```

---

## Python library usage

You can also use `IDXClient` directly in your own scripts:

```python
from idx_wrapper import IDXClient

client = IDXClient()

# Latest price for a single stock
bbca = client.get_stock_price("BBCA")
print(bbca)

# First 5 stocks from the full summary list
summary = client.get_stock_summary(length=5)
for stock in summary["data"]:
    print(stock)

# Daily prices for BBCA — January 2024
daily = client.get_stocks_daily(
    code="BBCA",
    date_from="2024-01-01",
    date_to="2024-01-31",
)
for row in daily["data"]:
    print(row)

# IHSG monthly index statistics
ihsg = client.get_index_statistics(index_id="COMPOSITE", period="monthly", length=5)
for row in ihsg["data"]:
    print(row)
```

Run the included sample script for a full demonstration:

```bash
python example.py
```

---

## Running tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## Project structure

```
IDX-py-wrapper/
├── idx_wrapper/
│   ├── __init__.py      # Package entry point
│   └── client.py        # IDXClient — all API calls
├── tests/
│   ├── test_client.py   # Unit tests for IDXClient
│   └── test_app.py      # Unit tests for FastAPI endpoints
├── app.py               # FastAPI server with sample endpoints
├── example.py           # Standalone usage examples
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Data source

All data is fetched from the public IDX APIs at **https://www.idx.co.id**.  
See [IDX Data Services](https://www.idx.co.id/en/products/idx-data-services/) for details.
