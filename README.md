# AML transaction risk scoring API

A FastAPI service that scores transactions for money-laundering risk. The model is
XGBoost, trained on the Elliptic AML dataset (203K transactions) with GPU-computed
graph features (cuDF, cuGraph), time-based splits and cost-sensitive learning.

**Results on the held-out test period:** PR-AUC 0.79, illicit F1 0.76,
100% precision on the top 500 alerts.

## Endpoints
- `GET /` : overview page
- `GET /docs` : interactive API documentation
- `POST /score` : score a transaction from its 169 features
- `GET /examples` : held-out test transactions you can score
- `GET /examples/{tx_index}/score` : live model inference on an example
- `GET /transaction/{tx_index}` : risk score plus transaction-network neighbourhood

## Run locally
```
pip install -r requirements.txt
uvicorn app:app --reload
```
