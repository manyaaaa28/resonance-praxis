# 🔮 RESONANCE
### *Predict Customer Behavior Before You Act*

[![Node.js](https://img.shields.io/badge/Node.js-18+-339933?style=flat-square&logo=node.js&logoColor=white)](https://nodejs.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Gemini](https://img.shields.io/badge/Google-Gemini%20AI-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-6D28D9?style=flat-square)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Built%20for-Praxis%202.0-FF6B6B?style=flat-square)](/)

---

> *"We don't just tell you that sales will drop — we show you **which customers will leave**, **why they're unhappy**, and **what specific price brings them back** — all by simulating thousands of AI Digital Twins in seconds."*

---

## 📖 Table of Contents

- [What is RESONANCE?](#-what-is-resonance)
- [The Problem We Solve](#-the-problem-we-solve)
- [Live Demo Scenario](#-live-demo-the-kentucky-shoes-scenario)
- [How It Works](#-how-it-works)
- [ML + GenAI Architecture](#-ml--genai-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [ML Model Details](#-ml-model-details)
- [API Reference](#-api-reference)
- [Hackathon Notes](#-hackathon-notes)

---

## 🔮 What is RESONANCE?

RESONANCE is a **Living Behavioral Sandbox** — a full-stack AI platform that lets business owners simulate real customer decision-making *before* making any real-world change.

Instead of risking revenue on untested pricing strategies, you:

1. **Clone** your customer base into AI-powered "Digital Twins"
2. **Configure** a What-If scenario (e.g., remove discounts, raise prices, test a new market)
3. **Watch** AI agents roleplay as your customers, sharing their internal thoughts in real time
4. **Decide** with confidence — zero real-world risk taken

```
  Your Data  →  ML Clustering  →  AI Personas  →  Simulation  →  Business Insight
    📊              🧠               🎭               ⚡               💡
```

---

## 🚨 The Problem We Solve

Every business owner faces the same terrifying question:

> *"If I change my pricing / stop promotions / enter a new market... what will actually happen?"*

| The Old Way 😰 | The RESONANCE Way ✅ |
|---|---|
| Test in real life, lose revenue if it fails | Simulate it — zero real cost |
| Wait 30 days for meaningful data | Get a prediction in **under 10 seconds** |
| Run expensive A/B tests | Unlimited What-If scenarios, free |
| Trust gut feeling | Ground decisions in data-driven agent behavior |
| Hire consultants for market research | AI-powered insights for any business owner |

---

## 🎯 Live Demo: The Kentucky Shoes Scenario

**The question:** *"If I stop giving discounts on Shoes in Kentucky, will people still buy them?"*

RESONANCE spins up four AI agents — one per customer segment — and simulates their reactions:

```
💰 Budget Hunter Agent (Kentucky, Male, 55):
   "I usually buy these every month, but without the promo code,
    it's not worth my Venmo balance. I found a similar pair $12
    cheaper elsewhere. Bring back even a 15% discount and I'll
    come back — but full price? No way."
   → ACTION: Left store. Switched brand.

💎 Loyal Customer Agent (Kentucky, Female, 42):
   "I've been buying from this brand for 3 years. I trust their
    quality. The price bump stings a little, but I'm not switching
    over it. I'll pay full price this time."
   → ACTION: Purchased. Used reward points.

👑 Premium Buyer Agent (Kentucky, Male, 48):
   "Full price is fine — I expect premium quality and I'm getting
    it. Actually, removing the discount makes it feel more exclusive."
   → ACTION: Purchased. Left 5-star review.

⚡ Impulse Shopper Agent (Kentucky, Female, 29):
   "Hmm, no deal right now. I'll put it in my wishlist and wait
    for Black Friday. Not urgent enough to buy at full price."
   → ACTION: Wishlisted. Did not convert.
```

**Instant Business Insight:**
Removing the discount causes the Budget segment (~27% of your base) to churn, and the Impulse segment to defer. Loyal and Premium customers stay regardless. **The optimal play:** keep the discount *only* for budget-sensitive customers via a targeted promo code — maximizing revenue while retaining at-risk buyers.

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                       RESONANCE PIPELINE                        │
├──────────────┬──────────────┬─────────────────┬────────────────┤
│  DATA LAYER  │   ML LAYER   │   AGENT LAYER   │  OUTPUT LAYER  │
│              │              │                 │                │
│ shopping_    │ K-Means      │ Gemini AI       │ Live Chat Feed │
│ trends.csv   │ Clustering   │ Agents          │                │
│              │              │                 │ Revenue        │
│ 3,900+       │ Logistic     │ Budget Hunter   │ Forecast       │
│ customer     │ Regression   │ Premium Buyer   │                │
│ records      │              │ Impulse Shopper │ Churn Risk     │
│              │ Random       │ Loyal Customer  │ Report         │
│              │ Forest       │                 │                │
│              │              │ Each agent gets │ Actionable     │
│              │ Revenue      │ a persona built │ Decision       │
│              │ Forecasting  │ from ML data    │                │
└──────────────┴──────────────┴─────────────────┴────────────────┘
```

### The 4-Step Flow

```
[Upload Dataset] → [ML Clusters Customers] → [Build AI Personas] → [Run Simulation] → [Decide]
       📊                    🧠                      🎭                    ⚡               💡
       
K-Means (k=4) identifies: Budget / Premium / Impulse / Loyal
Each cluster → system prompt → Gemini agent → human-readable reaction
```

---

## 🧠 ML + GenAI Architecture

RESONANCE is built on a **two-layer intelligence system**:

### Layer 1 — Machine Learning (The Brain)

> Python trains models offline → exports weights to JSON → Node.js loads at runtime. No Python required in production.

| Model | Purpose | Performance |
|---|---|---|
| **K-Means Clustering** (k=4) | Segment customers into 4 behavioral personas | Silhouette Score: 0.2339 |
| **Logistic Regression** | Purchase probability prediction | Accuracy: **98.33%** · F1: 0.98 |
| **Random Forest** | Churn risk classification | Accuracy: **78.59%** · F1: 0.77 |
| **Revenue Forecasting** | Expected value under discount scenarios | Probabilistic EV model |

**4 Behavioral Scores computed per customer:**

```python
price_sensitivity   = f(purchase_amount, discount_usage, promo_codes, shipping_type)
loyalty_score       = f(previous_purchases, subscription_status, review_rating, frequency)
impulse_score       = f(shipping_urgency, age, payment_speed, category)
quality_preference  = f(purchase_amount, premium_shipping, credit_card_usage)
```

These 4 scores → standardized → K-Means nearest-centroid → persona assignment.

### Layer 2 — Generative AI (The Voice)

> ML-grounded persona data → rich system prompt → Gemini agent → human-readable simulation

Each AI agent receives:

- 📍 **Demographics** — age, location, preferred payment method
- 📈 **Behavioral profile** — avg spend, purchase frequency, subscription status
- 🧠 **Psychological framing** — persona traits and decision-making patterns from ML clusters
- 🎯 **Scenario context** — the exact What-If being tested

**Result:** Cold data becomes human stories. Business owners read a chat feed — not a spreadsheet.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3, Vanilla JS — single-file, no build step needed |
| **Backend** | Node.js + Express.js |
| **ML Training** | Python, scikit-learn, pandas, numpy |
| **ML Runtime** | JavaScript (trained weights via JSON — no Python at runtime) |
| **GenAI** | Google Gemini API + Grok (xAI) via OpenAI-compatible SDK |
| **Streaming** | Server-Sent Events (SSE) for real-time agent thought delivery |
| **Database** | Supabase (PostgreSQL) for simulation history |
| **Dataset** | `shopping_trends.csv` — 3,900+ real customer records |

---

## 📁 Project Structure

```
resonance/
│
├── index.html                        # 🖥️  Full simulation UI (single-file frontend)
├── server.js                         # 🚀  Express API server + SSE streaming
├── package.json
├── vercel.json                       # ☁️  One-click Vercel deployment config
├── .env.example                      # 🔑  Environment variable template
│
├── src/
│   └── ml/
│       ├── predictiveModels.js       # 📊  Production ML inference (JS port of Python)
│       ├── personaClusterer.js       # 🎭  K-Means persona assignment at runtime
│       └── trained_model_weights.json # 💾  Exported scikit-learn model weights
│
├── images/                           # 🖼️  Product category images for UI
│
├── ml_model_development.ipynb        # 🔬  Full ML research + training notebook
└── train_models.py                   # 🐍  Retrain models + export weights
```

> **Key design choice:** ML models are trained in Python but run in JavaScript. This means zero Python dependency in production — the app deploys anywhere Node.js runs.

---

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- A Google Gemini API key (free tier works)
- Python 3.10+ *(only if retraining models)*

### 1. Clone the repo

```bash
git clone https://github.com/manyaaaa28/resonance-praxis.git
cd resonance-praxis
```

### 2. Install dependencies

```bash
npm install
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROK_API_KEY=your_grok_api_key_here        # optional fallback
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
PORT=3000
```

> **Get a free Gemini API key:** [aistudio.google.com](https://aistudio.google.com)

### 4. Start the server

```bash
npm start

# For development with auto-reload:
npm run dev
```

Open `http://localhost:3000` and run your first simulation. 🎉

---

### (Optional) Retrain ML Models

The repo ships with pre-trained weights so you can run immediately. To retrain on new data:

```bash
# Install Python dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Retrain and export weights to JSON
python train_models.py

# Or explore the full research interactively
jupyter notebook ml_model_development.ipynb
```

This regenerates `src/ml/trained_model_weights.json` with updated centroids and coefficients.

---

## 📊 ML Model Details

### Customer Segmentation — 4 Personas

| Persona | Count in Dataset | Avg Spend | Defining Trait |
|---|---|---|---|
| 💰 **Budget Hunter** | 1,049 customers | ~$35 | Price-sensitive, promo-driven, waits for deals |
| 👑 **Premium Buyer** | 1,029 customers | ~$85 | Quality-first, brand-conscious, discount-averse |
| ⚡ **Impulse Shopper** | 788 customers | ~$55 | Spontaneous, trend-driven, urgency-motivated |
| 💎 **Loyal Customer** | 1,034 customers | ~$65 | Repeat buyer, subscription holder, referral source |

### Logistic Regression — Feature Importance

The most influential features for purchase prediction:

```
price_sensitivity     →  +5.31  ← strongest signal overall
purchase_amount       →  +4.05
loyalty_score         →  +2.54
age                   →  +0.78
quality_preference    →  -1.96  (premium buyers respond to exclusivity, not discounts)
```

### Random Forest — Churn Risk Factors

```
review_rating         →  46.1%  importance  ← single most critical signal
loyalty_score         →  20.3%
previous_purchases    →  11.2%
purchase_amount       →   5.7%
```

> **Insight:** A customer's *review rating* is nearly 2.3× more predictive of churn than their *loyalty score*. A dissatisfied loyal customer is your highest churn risk.

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/health` | GET | Health check + model status |
| `POST /api/ml/forecast` | POST | Revenue forecast under a discount scenario |
| `POST /api/ml/elasticity` | POST | Price elasticity calculation across discount range |
| `POST /api/ml/churn` | POST | Churn risk score per persona |
| `GET /api/simulate/stream` | GET (SSE) | **Live-streaming** agent simulation |
| `POST /api/simulate` | POST | Single-shot simulation (non-streaming) |
| `GET /api/personas` | GET | Current K-Means cluster profiles |
| `POST /api/simulations` | POST | Save simulation result to database |
| `GET /api/simulations` | GET | Retrieve simulation history |

### Example — Run a Simulation

```bash
POST /api/simulate
Content-Type: application/json

{
  "scenario": "Remove shoe discounts in Kentucky",
  "parameters": {
    "category": "Shoes",
    "location": "Kentucky",
    "discount_pct": 0
  },
  "personas": ["budget", "loyal", "premium", "impulse"]
}
```

### Example — Response

```json
{
  "simulation_id": "sim_abc123",
  "revenue_impact": -18.4,
  "churn_risk": {
    "budget": 0.71,
    "loyal": 0.22,
    "premium": 0.09,
    "impulse": 0.44
  },
  "agent_responses": [
    {
      "persona": "budget",
      "agent_id": "Kentucky-Male-55",
      "thought": "Too expensive without the promo code. Found it cheaper elsewhere.",
      "action": "Left store"
    }
  ],
  "recommendation": "Apply targeted 15% discount to budget segment only to recover 71% churn risk"
}
```

### Streaming Simulation (SSE)

For the live agent chat experience, use the SSE endpoint:

```javascript
const params = new URLSearchParams({ discount: -20, category: 'Shoes', brand: 'premium', season: 'winter' });
const eventSource = new EventSource(`/api/simulate/stream?${params}`);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.type → 'thought' | 'metrics' | 'progress' | 'complete'
  if (data.type === 'thought') console.log(`${data.agentName}: ${data.thought}`);
};
```

---

## 🏆 Hackathon Notes

### Why RESONANCE is Different

Most prediction tools give you a number. RESONANCE gives you a **conversation.**

| Judging Criteria | How RESONANCE Delivers |
|---|---|
| ✅ **ML Integration** | K-Means + Logistic Regression + Random Forest — trained on 3,900 real records, not toy data |
| ✅ **GenAI Integration** | Gemini agents grounded in ML-derived persona data — not generic prompts |
| ✅ **Originality** | No other tool shows you *why* customers leave, in their own voice |
| ✅ **Real Data** | `shopping_trends.csv` — genuine behavioral dataset |
| ✅ **Production Ready** | Node.js + Vercel deploy config — runs in one command |
| ✅ **Reproducibility** | Judges can re-run `ml_model_development.ipynb` and verify every model metric |

### Suggested Demo Flow for Judges

```
Step 1 →  Open ml_model_development.ipynb
          See full ML training pipeline, silhouette scores, confusion matrices

Step 2 →  Open trained_model_weights.json
          Verify exported K-Means centroids and logistic regression coefficients

Step 3 →  Open personaClusterer.js
          See how JS consumes Python-trained weights at runtime

Step 4 →  Run the live app
          Simulate "remove discount on Winter Jackets in Mumbai"
          Watch agents react in real-time

Step 5 →  Change discount from -25% to 0%
          Watch different agents react differently
          Compare both scenarios side by side
```

### A Note on the Fallback Mode

If the Gemini API is unavailable (network issues at the venue), the app **automatically falls back** to a local simulation engine using pre-computed persona logic. The demo will still run — agent thoughts will still appear — the difference is they come from rule-based templates rather than live LLM calls.

---

## 🗺 Roadmap

- [x] Core simulation engine — 4 persona types
- [x] K-Means + Logistic Regression + Random Forest models
- [x] Gemini AI agent integration with SSE streaming
- [x] Compare Mode — side-by-side scenario analysis
- [x] Simulation history with persistence (Supabase)
- [ ] Custom dataset upload — train on your own customer data
- [ ] Multi-language persona support (Hindi, Tamil, Bengali)
- [ ] Industry-specific persona packs — retail, SaaS, hospitality, F&B
- [ ] Predictive ROI calculator with confidence intervals
- [ ] WhatsApp / Telegram bot interface for non-technical users

---

## 👥 Team

Built with ❤️ for **Praxis 2.0** by Team Resonance.

---

## 📄 License

[MIT](LICENSE) — free to use, remix, and build on.

---

<div align="center">

**RESONANCE** — *Fail 100 times in the simulation. Win once in the real world.*

⭐ If you found this useful, star the repo!

</div>
