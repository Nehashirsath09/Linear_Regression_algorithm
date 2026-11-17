import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(os.path.dirname(__file__), "linear.pkl")
)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HomeWorth · House Price Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0b1020; --card:rgba(255,255,255,.06); --border:rgba(255,255,255,.12);
    --text:#eef1ff; --muted:#a3abc7; --accent:#7c9cff; --accent2:#b583ff; --good:#4ade80;
  }
  *{box-sizing:border-box;margin:0}
  body{
    font-family:Inter,system-ui,sans-serif;color:var(--text);min-height:100vh;
    background:
      radial-gradient(900px 500px at 10% -10%,rgba(124,156,255,.28),transparent 60%),
      radial-gradient(800px 500px at 100% 10%,rgba(181,131,255,.22),transparent 60%),
      var(--bg);
    padding:40px 16px 60px;
  }
  .wrap{max-width:980px;margin:0 auto}
  header{text-align:center;margin-bottom:36px}
  .logo{display:inline-flex;align-items:center;gap:10px;font-weight:600;color:var(--muted);
    border:1px solid var(--border);padding:6px 14px;border-radius:99px;font-size:13px;background:var(--card)}
  h1{font-family:Fraunces,serif;font-size:clamp(34px,6vw,56px);line-height:1.1;margin:18px 0 10px;
    background:linear-gradient(90deg,#fff,var(--accent) 60%,var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent}
  header p{color:var(--muted);max-width:520px;margin:0 auto;line-height:1.55}
  .grid{display:grid;grid-template-columns:1.35fr 1fr;gap:22px}
  @media(max-width:820px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--border);border-radius:22px;padding:26px;
    backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);box-shadow:0 20px 50px rgba(0,0,0,.35)}
  .card h2{font-size:15px;font-weight:600;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;margin-bottom:18px}
  .fields{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:520px){.fields{grid-template-columns:1fr}}
  .field.full{grid-column:1/-1}
  label{display:block;font-size:13px;color:var(--muted);margin-bottom:6px;font-weight:500}
  .input{position:relative}
  .input span{position:absolute;left:12px;top:50%;transform:translateY(-50%);font-size:16px}
  input[type=number]{
    width:100%;padding:12px 14px 12px 38px;border-radius:12px;border:1px solid var(--border);
    background:rgba(0,0,0,.25);color:var(--text);font:inherit;font-size:15px;outline:none;transition:.2s}
  input[type=number]:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(124,156,255,.25)}
  input[type=range]{width:100%;accent-color:var(--accent)}
  .qrow{display:flex;justify-content:space-between;align-items:center}
  .qval{font-weight:600;color:var(--accent)}
  button{
    width:100%;margin-top:22px;padding:15px;border:0;border-radius:14px;font:inherit;font-weight:600;font-size:16px;
    color:#fff;cursor:pointer;background:linear-gradient(90deg,var(--accent),var(--accent2));
    box-shadow:0 10px 30px rgba(124,156,255,.35);transition:transform .15s,box-shadow .15s}
  button:hover{transform:translateY(-2px);box-shadow:0 14px 36px rgba(124,156,255,.5)}
  button:disabled{opacity:.6;cursor:wait;transform:none}
  .result{display:flex;flex-direction:column;justify-content:center;text-align:center;min-height:260px}
  .house{font-size:54px;margin-bottom:6px}
  .label{color:var(--muted);font-size:13px;letter-spacing:.06em;text-transform:uppercase}
  .price{font-family:Fraunces,serif;font-size:clamp(36px,6vw,50px);font-weight:700;margin:8px 0;
    background:linear-gradient(90deg,var(--good),#a7f3d0);-webkit-background-clip:text;background-clip:text;color:transparent}
  .hint{color:var(--muted);font-size:14px;line-height:1.5}
  .placeholder .price{background:none;color:rgba(255,255,255,.25);-webkit-text-fill-color:initial}
  .err{color:#fca5a5}
  .pop{animation:pop .5s ease}
  @keyframes pop{from{transform:scale(.92);opacity:0}to{transform:scale(1);opacity:1}}
  footer{text-align:center;color:var(--muted);font-size:12px;margin-top:30px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">🏡 HomeWorth · Linear Regression</div>
    <h1>What's your home worth?</h1>
    <p>Enter a few details about the property and get an instant machine-learning price estimate.</p>
  </header>

  <div class="grid">
    <form class="card" id="form">
      <h2>Property details</h2>
      <div class="fields">
        <div class="field"><label>Square footage</label>
          <div class="input"><span>📐</span><input type="number" name="Square_Footage" value="2000" min="100" step="any" required></div></div>
        <div class="field"><label>Year built</label>
          <div class="input"><span>📅</span><input type="number" name="Year_Built" value="2005" min="1800" max="2100" required></div></div>
        <div class="field"><label>Bedrooms</label>
          <div class="input"><span>🛏️</span><input type="number" name="Num_Bedrooms" value="3" min="0" step="1" required></div></div>
        <div class="field"><label>Bathrooms</label>
          <div class="input"><span>🛁</span><input type="number" name="Num_Bathrooms" value="2" min="0" step="1" required></div></div>
        <div class="field"><label>Lot size</label>
          <div class="input"><span>🌳</span><input type="number" name="Lot_Size" value="1.5" min="0" step="any" required></div></div>
        <div class="field"><label>Garage size (cars)</label>
          <div class="input"><span>🚗</span><input type="number" name="Garage_Size" value="2" min="0" step="1" required></div></div>
        <div class="field full">
          <div class="qrow"><label>Neighborhood quality</label><span class="qval" id="qv">5 / 10</span></div>
          <input type="range" name="Neighborhood_Quality" id="q" min="1" max="10" value="5">
        </div>
      </div>
      <button type="submit" id="btn">Predict price ✨</button>
    </form>

    <div class="card result placeholder" id="res">
      <div class="house">🏠</div>
      <div class="label">Estimated value</div>
      <div class="price" id="price">$ —</div>
      <div class="hint" id="hint">Fill in the details and press <b>Predict</b>.</div>
    </div>
  </div>
  <footer>Built with Flask &amp; scikit-learn · Estimates are for illustration only.</footer>
</div>

<script>
const form = document.getElementById('form'), res = document.getElementById('res'),
      price = document.getElementById('price'), hint = document.getElementById('hint'),
      btn = document.getElementById('btn'), q = document.getElementById('q');
q.addEventListener('input', () => document.getElementById('qv').textContent = q.value + ' / 10');

form.addEventListener('submit', async e => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(form).entries());
  btn.disabled = true; btn.textContent = 'Calculating…';
  try {
    const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Prediction failed');
    res.classList.remove('placeholder'); price.classList.remove('err');
    price.textContent = j.formatted;
    hint.textContent = 'Based on ' + Number(data.Square_Footage).toLocaleString() + ' sq ft, ' +
      data.Num_Bedrooms + ' bed / ' + data.Num_Bathrooms + ' bath, built ' + data.Year_Built + '.';
    res.classList.remove('pop'); void res.offsetWidth; res.classList.add('pop');
  } catch (err) {
    price.textContent = 'Oops'; price.classList.add('err'); hint.textContent = err.message;
  } finally {
    btn.disabled = false; btn.textContent = 'Predict price ✨';
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        row = {f: float(payload[f]) for f in FEATURES}
    except KeyError as e:
        return jsonify(error=f"Missing field: {e.args[0]}"), 400
    except (TypeError, ValueError):
        return jsonify(error="All fields must be valid numbers."), 400

    X = pd.DataFrame([row], columns=FEATURES)
    pred = float(model.predict(X).ravel()[0])
    return jsonify(prediction=pred, formatted=f"${pred:,.0f}")


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
