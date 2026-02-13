"""
RESONANCE - ML Model Training Script

Trains real scikit-learn models on customer data and exports
learned parameters to JSON for use by the Node.js production backend.

Models trained:
1. Logistic Regression - Purchase probability prediction
2. K-Means Clustering  - Customer segmentation (4 personas)
3. Random Forest        - Churn risk prediction

Usage:
    cd resonance/src/ml
    python train_models.py
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================

def load_and_engineer_features(csv_path):
    """Load CSV and create behavioral features."""
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- Price Sensitivity Score (0-1) ---
    df['price_sensitivity'] = 0.0
    df.loc[df['Purchase Amount (USD)'] < 40, 'price_sensitivity'] += 0.3
    df.loc[(df['Purchase Amount (USD)'] >= 40) & (df['Purchase Amount (USD)'] < 60), 'price_sensitivity'] += 0.15
    df.loc[df['Discount Applied'] == 'Yes', 'price_sensitivity'] += 0.2
    df.loc[df['Promo Code Used'] == 'Yes', 'price_sensitivity'] += 0.2
    df.loc[df['Shipping Type'] == 'Free Shipping', 'price_sensitivity'] += 0.15
    df.loc[df['Shipping Type'] == 'Standard', 'price_sensitivity'] += 0.1
    df['price_sensitivity'] = df['price_sensitivity'].clip(0, 1)

    # --- Loyalty Score (0-1) ---
    df['loyalty_score'] = 0.0
    df.loc[df['Previous Purchases'] > 40, 'loyalty_score'] += 0.35
    df.loc[(df['Previous Purchases'] > 25) & (df['Previous Purchases'] <= 40), 'loyalty_score'] += 0.2
    df.loc[(df['Previous Purchases'] > 10) & (df['Previous Purchases'] <= 25), 'loyalty_score'] += 0.1
    df.loc[df['Subscription Status'] == 'Yes', 'loyalty_score'] += 0.25
    df.loc[df['Review Rating'] >= 4.5, 'loyalty_score'] += 0.2
    df.loc[(df['Review Rating'] >= 4.0) & (df['Review Rating'] < 4.5), 'loyalty_score'] += 0.1
    df['loyalty_score'] = df['loyalty_score'].clip(0, 1)

    # --- Impulse Score (0-1) ---
    df['impulse_score'] = 0.0
    df.loc[df['Shipping Type'] == 'Next Day Air', 'impulse_score'] += 0.3
    df.loc[df['Shipping Type'] == 'Express', 'impulse_score'] += 0.2
    df.loc[df['Age'] < 30, 'impulse_score'] += 0.2
    df.loc[(df['Age'] >= 30) & (df['Age'] < 40), 'impulse_score'] += 0.1
    df.loc[df['Category'] == 'Accessories', 'impulse_score'] += 0.15
    df['impulse_score'] = df['impulse_score'].clip(0, 1)

    # --- Quality Preference Score (0-1) ---
    df['quality_preference'] = 0.0
    df.loc[df['Purchase Amount (USD)'] > 80, 'quality_preference'] += 0.3
    df.loc[(df['Purchase Amount (USD)'] > 60) & (df['Purchase Amount (USD)'] <= 80), 'quality_preference'] += 0.15
    df.loc[df['Review Rating'] >= 4.5, 'quality_preference'] += 0.15
    df.loc[df['Age'] > 45, 'quality_preference'] += 0.1
    df.loc[df['Discount Applied'] == 'No', 'quality_preference'] += 0.1
    df['quality_preference'] = df['quality_preference'].clip(0, 1)

    print(f"✅ Engineered 4 behavioral features")
    return df


# ============================================================
# 2. MODEL 1: LOGISTIC REGRESSION (Purchase Probability)
# ============================================================

def train_purchase_model(df):
    """
    Train Logistic Regression to predict purchase probability.
    
    Target: Whether customer would buy given a discount scenario.
    We synthesize the target by combining behavioral signals:
    - Customers who used discounts AND have high frequency = likely buyers
    """
    print("\n" + "=" * 60)
    print("Training Model 1: Purchase Probability (Logistic Regression)")
    print("=" * 60)

    # Create synthetic purchase target based on real behavioral signals
    # A customer is likely to "buy" if they show strong positive signals
    df['would_buy'] = (
        (df['Discount Applied'] == 'Yes').astype(int) * 0.3 +
        (df['Previous Purchases'] > 15).astype(int) * 0.25 +
        (df['Review Rating'] >= 3.5).astype(int) * 0.2 +
        (df['Subscription Status'] == 'Yes').astype(int) * 0.15 +
        (df['Promo Code Used'] == 'Yes').astype(int) * 0.1
    )
    df['buy_label'] = (df['would_buy'] > 0.45).astype(int)

    # Features for the model
    feature_cols = [
        'price_sensitivity', 'loyalty_score', 'impulse_score',
        'quality_preference', 'Age', 'Previous Purchases',
        'Review Rating', 'Purchase Amount (USD)'
    ]

    X = df[feature_cols].values
    y = df['buy_label'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Coefficients: {model.coef_[0].tolist()}")
    print(f"  Intercept: {model.intercept_[0]:.4f}")

    # Map coefficients to feature names for readability
    coef_dict = {}
    for i, col in enumerate(feature_cols):
        coef_dict[col] = round(float(model.coef_[0][i]), 6)

    # Also export scaler params for normalization in JS
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }

    return {
        'model_type': 'logistic_regression',
        'feature_names': feature_cols,
        'coefficients': coef_dict,
        'intercept': round(float(model.intercept_[0]), 6),
        'scaler': scaler_params,
        'metrics': {
            'accuracy': round(accuracy, 4),
            'f1_score': round(f1, 4),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    }


# ============================================================
# 3. MODEL 2: K-MEANS CLUSTERING (Customer Segmentation)
# ============================================================

def train_clustering_model(df):
    """
    Train K-Means (k=4) to discover natural customer segments.
    Replaces hardcoded threshold-based segmentation.
    """
    print("\n" + "=" * 60)
    print("Training Model 2: Customer Segmentation (K-Means, k=4)")
    print("=" * 60)

    # Clustering features
    cluster_features = [
        'price_sensitivity', 'loyalty_score',
        'impulse_score', 'quality_preference'
    ]

    X = df[cluster_features].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)

    # Evaluate
    labels = kmeans.labels_
    sil_score = silhouette_score(X_scaled, labels)
    print(f"  Silhouette Score: {sil_score:.4f}")

    # Get centroids (in original scale)
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    # Map clusters to persona types based on centroid characteristics
    # Find which cluster best matches each persona type
    persona_mapping = {}
    cluster_profiles = {}

    for i in range(4):
        centroid = centroids_original[i]
        profile = {
            'price_sensitivity': round(float(centroid[0]), 4),
            'loyalty_score': round(float(centroid[1]), 4),
            'impulse_score': round(float(centroid[2]), 4),
            'quality_preference': round(float(centroid[3]), 4)
        }
        cluster_profiles[i] = profile

    # Assign persona types by dominant trait in each cluster
    # budget = highest price_sensitivity
    # premium = highest quality_preference
    # impulse = highest impulse_score
    # loyal = highest loyalty_score
    traits = ['price_sensitivity', 'loyalty_score', 'impulse_score', 'quality_preference']
    persona_types = ['budget', 'loyal', 'impulse', 'premium']

    used_clusters = set()
    for trait_idx, persona_type in enumerate(persona_types):
        trait_name = traits[trait_idx]
        # Find cluster with highest value for this trait (excluding already assigned)
        best_cluster = None
        best_value = -1
        for c_id, profile in cluster_profiles.items():
            if c_id not in used_clusters and profile[trait_name] > best_value:
                best_value = profile[trait_name]
                best_cluster = c_id
        persona_mapping[persona_type] = best_cluster
        used_clusters.add(best_cluster)

    print(f"  Cluster → Persona mapping: {persona_mapping}")

    # Count customers per cluster
    cluster_counts = {}
    for persona_type, cluster_id in persona_mapping.items():
        count = int(np.sum(labels == cluster_id))
        cluster_counts[persona_type] = count
        print(f"    {persona_type}: {count} customers (cluster {cluster_id})")

    # Prepare centroids mapped to persona names
    persona_centroids = {}
    for persona_type, cluster_id in persona_mapping.items():
        persona_centroids[persona_type] = {
            'centroid': {
                'price_sensitivity': round(float(centroids_original[cluster_id][0]), 4),
                'loyalty_score': round(float(centroids_original[cluster_id][1]), 4),
                'impulse_score': round(float(centroids_original[cluster_id][2]), 4),
                'quality_preference': round(float(centroids_original[cluster_id][3]), 4)
            },
            'centroid_scaled': centroids_scaled[cluster_id].tolist(),
            'count': cluster_counts[persona_type]
        }

    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }

    return {
        'model_type': 'kmeans',
        'k': 4,
        'feature_names': cluster_features,
        'persona_centroids': persona_centroids,
        'scaler': scaler_params,
        'cluster_to_persona': {str(v): k for k, v in persona_mapping.items()},
        'metrics': {
            'silhouette_score': round(sil_score, 4),
            'total_customers': len(df),
            'cluster_sizes': cluster_counts
        }
    }


# ============================================================
# 4. MODEL 3: RANDOM FOREST (Churn Prediction)
# ============================================================

def train_churn_model(df):
    """
    Train Random Forest to predict customer churn risk.
    """
    print("\n" + "=" * 60)
    print("Training Model 3: Churn Risk Prediction (Random Forest)")
    print("=" * 60)

    # Create churn label from behavioral signals
    # High churn risk if: low frequency + low rating + no subscription + few purchases
    freq_map = {
        'Weekly': 0, 'Bi-Weekly': 0, 'Fortnightly': 0,
        'Monthly': 0.1, 'Every 3 Months': 0.2,
        'Quarterly': 0.3, 'Annually': 0.5
    }
    df['freq_risk'] = df['Frequency of Purchases'].map(freq_map).fillna(0.2)

    df['churn_score'] = (
        df['freq_risk'] +
        (df['Review Rating'] < 3.5).astype(float) * 0.25 +
        (df['Subscription Status'] == 'No').astype(float) * 0.15 +
        (df['Previous Purchases'] < 10).astype(float) * 0.15
    )
    df['churn_label'] = (df['churn_score'] > 0.35).astype(int)

    # Features
    feature_cols = [
        'price_sensitivity', 'loyalty_score', 'impulse_score',
        'quality_preference', 'Age', 'Previous Purchases',
        'Review Rating', 'Purchase Amount (USD)'
    ]

    X = df[feature_cols].values
    y = df['churn_label'].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        min_samples_split=5
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Feature importances
    importances = {}
    for i, col in enumerate(feature_cols):
        importances[col] = round(float(model.feature_importances_[i]), 6)

    print(f"  Feature importances: {importances}")

    # Export decision tree thresholds from the first few trees 
    # for JS approximation using feature importance as weights
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }

    # Also get the model's predicted probabilities pattern
    # by testing on representative samples to create a lookup-style model
    churn_thresholds = {
        'high_risk_threshold': 0.6,
        'medium_risk_threshold': 0.3
    }

    # Get average prediction probabilities by feature ranges for JS approximation
    test_probs = model.predict_proba(X_scaled)[:, 1]
    
    # Create bins for approximation
    bins = {
        'low_loyalty_high_churn': float(np.mean(test_probs[df['loyalty_score'] < 0.3])),
        'high_loyalty_low_churn': float(np.mean(test_probs[df['loyalty_score'] > 0.6])),
        'low_frequency_churn': float(np.mean(test_probs[df['freq_risk'] > 0.3])),
        'high_frequency_churn': float(np.mean(test_probs[df['freq_risk'] < 0.1])),
        'low_rating_churn': float(np.mean(test_probs[df['Review Rating'] < 3.5])),
        'high_rating_churn': float(np.mean(test_probs[df['Review Rating'] >= 4.0])),
    }

    return {
        'model_type': 'random_forest',
        'feature_names': feature_cols,
        'feature_importances': importances,
        'scaler': scaler_params,
        'thresholds': churn_thresholds,
        'probability_bins': {k: round(v, 4) for k, v in bins.items()},
        'metrics': {
            'accuracy': round(accuracy, 4),
            'f1_score': round(f1, 4),
            'n_estimators': 100,
            'max_depth': 8,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    }


# ============================================================
# 5. MAIN: TRAIN ALL & EXPORT
# ============================================================

def main():
    print("🔮 RESONANCE - ML Model Training Pipeline")
    print("=" * 60)

    # Find CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'shopping_trends.csv')

    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at: {csv_path}")
        return

    # Load & engineer features
    df = load_and_engineer_features(csv_path)

    # Train all 3 models
    purchase_model = train_purchase_model(df)
    clustering_model = train_clustering_model(df)
    churn_model = train_churn_model(df)

    # Combine all model outputs
    trained_models = {
        'metadata': {
            'trained_at': pd.Timestamp.now().isoformat(),
            'dataset_rows': len(df),
            'dataset_file': 'shopping_trends.csv',
            'framework': 'scikit-learn',
            'python_version': '3.x'
        },
        'logistic_regression': purchase_model,
        'kmeans': clustering_model,
        'random_forest': churn_model
    }

    # Export to JSON
    output_path = os.path.join(script_dir, 'trained_model_weights.json')
    with open(output_path, 'w') as f:
        json.dump(trained_models, f, indent=2)

    print("\n" + "=" * 60)
    print("🎉 ALL MODELS TRAINED & EXPORTED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n📁 Output: {output_path}")
    print(f"\n📊 Summary:")
    print(f"  Model 1 (Logistic Regression): Accuracy={purchase_model['metrics']['accuracy']}, F1={purchase_model['metrics']['f1_score']}")
    print(f"  Model 2 (K-Means):             Silhouette={clustering_model['metrics']['silhouette_score']}")
    print(f"  Model 3 (Random Forest):       Accuracy={churn_model['metrics']['accuracy']}, F1={churn_model['metrics']['f1_score']}")
    print(f"\n💾 Weights exported to trained_model_weights.json")
    print(f"🔗 Ready for Node.js backend integration!")


if __name__ == '__main__':
    main()
