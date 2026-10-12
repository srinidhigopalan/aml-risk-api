import json, time
import numpy as np, pandas as pd, xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

META = json.load(open('model_meta.json'))
FEATURES, THRESHOLD = META['features'], META['threshold']
MODEL = xgb.Booster()
MODEL.load_model('model.json')          # plain Booster: no scikit-learn needed
SCORES = pd.read_parquet('scores.parquet').set_index('idx')
EDGES = pd.read_parquet('edges_idx.parquet')
EXAMPLES = {e['tx_index']: e for e in json.load(open('examples.json'))}
LABELS = {-1: 'unknown', 0: 'licit', 1: 'illicit'}

app = FastAPI(title='AML transaction risk scoring API',
              description='XGBoost fraud model trained on the Elliptic AML dataset with GPU graph features.')


class Transaction(BaseModel):
    features: list[float]


def predict(features):
    if len(features) != len(FEATURES):
        raise HTTPException(422, f'Expected {len(FEATURES)} features, got {len(features)}')
    start = time.perf_counter()
    p = float(MODEL.predict(xgb.DMatrix(np.array([features], dtype=np.float32)))[0])
    return {'risk_score': round(p, 6), 'flagged': p >= THRESHOLD, 'threshold': round(THRESHOLD, 3),
            'model_ms': round((time.perf_counter() - start) * 1000, 2)}


@app.get('/', response_class=HTMLResponse)
def home():
    ex = next(iter(EXAMPLES))
    return f"""<html><head><title>AML risk scoring API</title></head>
    <body style="font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;line-height:1.6">
    <h1>AML transaction risk scoring API</h1>
    <p>An XGBoost model that scores transactions for money-laundering risk, trained on the
    Elliptic AML dataset (203K transactions) with GPU-computed graph features.</p>
    <ul>
      <li><a href="/docs">Interactive API docs</a>: try every endpoint in your browser</li>
      <li><a href="/examples">Example test transactions</a></li>
      <li><a href="/examples/{ex}/score">Live score for example {ex}</a></li>
      <li><a href="/transaction/{ex}">Network view of transaction {ex}</a></li>
    </ul></body></html>"""


@app.get('/health')
def health():
    return {'status': 'ok', 'n_features': len(FEATURES), 'n_transactions': len(SCORES)}


@app.post('/score')
def score(t: Transaction):
    return predict(t.features)


@app.get('/examples')
def examples():
    return [{'tx_index': k, 'true_label': LABELS[e['true_label']],
             'score_url': f'/examples/{k}/score'} for k, e in EXAMPLES.items()]


@app.get('/examples/{tx_index}/score')
def score_example(tx_index: int):
    if tx_index not in EXAMPLES:
        raise HTTPException(404, 'Not an example transaction; see /examples')
    e = EXAMPLES[tx_index]
    return {'tx_index': tx_index, 'true_label': LABELS[e['true_label']], **predict(e['features'])}


@app.get('/transaction/{tx_index}')
def transaction(tx_index: int):
    if tx_index not in SCORES.index:
        raise HTTPException(404, f'tx_index must be between 0 and {len(SCORES) - 1}')
    r = SCORES.loc[tx_index]
    nb = EDGES[EDGES.src == tx_index].dst.tolist()
    nbs = SCORES.loc[nb] if nb else SCORES.iloc[0:0]
    return {'tx_index': tx_index, 'time_step': int(r.time_step), 'known_label': LABELS[int(r.label)],
            'xgb_risk_score': round(float(r.xgb_score), 6), 'gnn_risk_score': round(float(r.gnn_score), 6),
            'flagged': bool(r.xgb_score >= THRESHOLD),
            'graph': {'in_degree': int(r.in_degree), 'out_degree': int(r.out_degree),
                      'community_size': int(r.community_size)},
            'neighbours': {'count': len(nb), 'flagged': int((nbs.xgb_score >= THRESHOLD).sum()),
                           'known_illicit': int((nbs.label == 1).sum()),
                           'top': [{'tx_index': int(i), 'xgb_risk_score': round(float(s), 4)}
                                   for i, s in nbs.xgb_score.sort_values(ascending=False).head(5).items()]}}
