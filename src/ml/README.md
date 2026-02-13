# ML Models - Development & Integration

## 📓 Jupyter Notebook: `ml_model_development.ipynb`

This notebook contains the complete ML model development process with data analysis, visualization, model training, and evaluation.

### What's Inside:

1. **Data Loading & Exploration**
   - Load `shopping_trends.csv` (3000+ customer records)
   - Exploratory data analysis with visualizations
   - Statistical summaries

2. **Feature Engineering**
   - Price sensitivity scoring
   - Loyalty score calculation
   - Impulse buying detection
   - Quality preference analysis

3. **Customer Segmentation**
   - Clustering into 4 personas: Budget, Premium, Impulse, Loyal
   - Visualizations of segment characteristics

4. **ML Models**
   - **Purchase Probability Predictor** (weighted feature scoring)
   - **Revenue Forecasting Model** (probabilistic expected value)
   - **Churn Prediction Model** (risk-based classification)
   - **Price Elasticity Calculator** (demand sensitivity analysis)

5. **Model Evaluation & Visualizations**
   - Performance metrics
   - Charts saved to `images/` directory

---

## 🔗 Integration with Node.js Backend

The models developed in the notebook are implemented in production-ready JavaScript code at:

**`src/ml/predictiveModels.js`**

### How It Works:

```
┌─────────────────────────┐
│  ml_model_development   │
│        .ipynb           │
│  (Python - Research)    │
│                         │
│  • Data exploration     │
│  • Model prototyping    │
│  • Visualization        │
│  • Algorithm tuning     │
└───────────┬─────────────┘
            │
            │ Algorithm Transfer
            ▼
┌─────────────────────────┐
│   predictiveModels.js   │
│  (JavaScript - Prod)    │
│                         │
│  • Same algorithms      │
│  • Optimized for speed  │
│  • API integration      │
│  • Real-time scoring    │
└───────────┬─────────────┘
            │
            │ API Endpoints
            ▼
┌─────────────────────────┐
│     Resonance API       │
│                         │
│  • /api/ml/forecast     │
│  • /api/ml/elasticity   │
│  • /api/ml/churn        │
└─────────────────────────┘
```

---

## 🚀 Running the Notebook

### Prerequisites:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Start Jupyter:
```bash
jupyter notebook ml_model_development.ipynb
```

### Run All Cells:
- This will generate all visualizations and save them to `images/`
- Model summary exported to `ml_model_summary.json`

---

## 📊 Generated Outputs

After running the notebook, you'll have:

1. **`images/customer_segments.png`** - Customer segmentation visualization
2. **`images/purchase_probability_dist.png`** - Probability distribution chart
3. **`images/revenue_forecast.png`** - Revenue vs discount analysis
4. **`images/churn_risk_analysis.png`** - Churn risk distribution
5. **`images/price_elasticity.png`** - Price elasticity curves
6. **`ml_model_summary.json`** - Model performance metrics

---

## 🎯 For Hackathon Judges

**Why This Approach?**

1. ✅ **Clear ML Development** - Jupyter notebook shows full ML workflow
2. ✅ **Production Integration** - Models deployed in real backend
3. ✅ **Reproducibility** - Judges can re-run the notebook
4. ✅ **Visualizations** - Professional charts demonstrate insights
5. ✅ **Real Data** - Uses actual CSV dataset (not toy data)

**Demo Flow:**

1. Show `ml_model_development.ipynb` → ML development process
2. Show `src/ml/predictiveModels.js` → Production implementation
3. Show `server.js` → API endpoints using the models
4. Demo the live app → ML predictions in action

This demonstrates **full-stack ML engineering** - from research to production! 🚀
