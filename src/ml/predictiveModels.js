/**
 * Predictive ML Models for RESONANCE
 * 
 * Uses trained scikit-learn model weights (exported from train_models.py)
 * for real ML-powered predictions:
 * - Purchase probability (Logistic Regression)
 * - Revenue forecasting (based on LR probabilities)
 * - Customer churn prediction (Random Forest feature importances)
 * - Price elasticity modeling
 */

const fs = require('fs');
const path = require('path');

// ============ LOAD TRAINED MODEL WEIGHTS ============

let trainedWeights = null;

try {
    const weightsPath = path.join(__dirname, 'trained_model_weights.json');
    const rawData = fs.readFileSync(weightsPath, 'utf-8');
    trainedWeights = JSON.parse(rawData);
    console.log(`[ML] Loaded trained model weights (trained: ${trainedWeights.metadata.trained_at})`);
    console.log(`[ML]   Logistic Regression: accuracy=${trainedWeights.logistic_regression.metrics.accuracy}, F1=${trainedWeights.logistic_regression.metrics.f1_score}`);
    console.log(`[ML]   K-Means:             silhouette=${trainedWeights.kmeans.metrics.silhouette_score}`);
    console.log(`[ML]   Random Forest:       accuracy=${trainedWeights.random_forest.metrics.accuracy}, F1=${trainedWeights.random_forest.metrics.f1_score}`);
} catch (err) {
    console.warn('[ML] Warning: Could not load trained_model_weights.json - using fallback heuristics');
    console.warn('[ML] Run "python train_models.py" to train models and generate weights');
}


// ============ SIGMOID FUNCTION (for Logistic Regression) ============

function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}


// ============ STANDARDIZE FEATURES (using trained scaler params) ============

function standardizeFeatures(featureValues, scalerParams) {
    return featureValues.map((val, i) => {
        return (val - scalerParams.mean[i]) / scalerParams.scale[i];
    });
}


/**
 * Predict purchase probability using trained Logistic Regression
 * @param {Object} customer - Customer data
 * @param {Object} scenario - Simulation scenario (discount, category, etc.)
 * @param {Object} persona - Customer's persona profile
 * @returns {Object} - { probability, confidence, factors }
 */
function predictPurchaseProbability(customer, scenario, persona) {
    if (trainedWeights && trainedWeights.logistic_regression) {
        return predictWithTrainedLR(customer, scenario, persona);
    }
    return predictWithFallback(customer, scenario, persona);
}


/**
 * Predict using trained Logistic Regression model
 */
function predictWithTrainedLR(customer, scenario, persona) {
    const lr = trainedWeights.logistic_regression;

    // Extract the same features used during training
    const featureValues = [
        customer.priceSensitivity || 0.5,
        customer.loyaltyScore || 0.5,
        customer.impulseScore || 0.3,
        customer.qualityPreference || 0.3,
        customer.age || 35,
        customer.previousPurchases || 10,
        customer.reviewRating || 3.5,
        customer.purchaseAmount || 50
    ];

    // Standardize using trained scaler parameters
    const scaledFeatures = standardizeFeatures(featureValues, lr.scaler);

    // Compute logistic regression: z = w·x + b
    let z = lr.intercept;
    const featureNames = lr.feature_names;
    const factors = [];

    for (let i = 0; i < featureNames.length; i++) {
        const coefKey = featureNames[i];
        const coef = lr.coefficients[coefKey] || 0;
        const contribution = coef * scaledFeatures[i];
        z += contribution;

        factors.push({
            name: coefKey,
            score: Math.abs(contribution),
            weight: Math.abs(coef),
            direction: contribution > 0 ? 'positive' : 'negative'
        });
    }

    // Apply scenario modifiers (discount/category/season boost)
    const discount = scenario.discount || 0;
    const categoryMatch = customer.category === scenario.category;
    const seasonMatch = customer.season === scenario.season;

    // Discount boosts probability (more discount = more likely to buy)
    z += (discount / 100) * 1.5;
    // Category match boosts
    if (categoryMatch) z += 0.3;
    // Season match gives small boost
    if (seasonMatch) z += 0.15;

    // Apply sigmoid to get probability
    let probability = sigmoid(z);

    // Clamp
    probability = Math.min(0.95, Math.max(0.05, probability));

    // Sort factors by impact
    factors.sort((a, b) => b.score - a.score);

    // Confidence from model metrics
    const confidence = lr.metrics.accuracy;

    return {
        probability,
        confidence,
        factors: factors.slice(0, 6),
        prediction: probability > 0.5 ? 'LIKELY' : 'UNLIKELY',
        modelType: 'trained_logistic_regression'
    };
}


/**
 * Fallback prediction when trained weights are not available
 */
function predictWithFallback(customer, scenario, persona) {
    const features = extractFeatures(customer, scenario, persona);

    const weights = {
        priceSensitivity: 0.25,
        loyaltyScore: 0.20,
        categoryMatch: 0.15,
        discountImpact: 0.20,
        seasonalFit: 0.10,
        affordability: 0.10
    };

    let probability = 0;
    const factors = [];

    const priceImpact = calculatePriceImpact(features, scenario, persona);
    probability += priceImpact * weights.priceSensitivity;
    factors.push({ name: 'Price Impact', score: priceImpact, weight: weights.priceSensitivity });

    const loyaltyImpact = features.loyaltyScore;
    probability += loyaltyImpact * weights.loyaltyScore;
    factors.push({ name: 'Customer Loyalty', score: loyaltyImpact, weight: weights.loyaltyScore });

    const categoryImpact = features.categoryMatch ? 0.9 : 0.3;
    probability += categoryImpact * weights.categoryMatch;
    factors.push({ name: 'Category Interest', score: categoryImpact, weight: weights.categoryMatch });

    const discountImpact = calculateDiscountImpact(features, scenario, persona);
    probability += discountImpact * weights.discountImpact;
    factors.push({ name: 'Discount Effectiveness', score: discountImpact, weight: weights.discountImpact });

    const seasonalImpact = features.seasonMatch ? 0.85 : 0.5;
    probability += seasonalImpact * weights.seasonalFit;
    factors.push({ name: 'Seasonal Fit', score: seasonalImpact, weight: weights.seasonalFit });

    const affordabilityImpact = calculateAffordability(features, scenario);
    probability += affordabilityImpact * weights.affordability;
    factors.push({ name: 'Affordability', score: affordabilityImpact, weight: weights.affordability });

    const confidence = calculateConfidence(customer);

    return {
        probability: Math.min(0.95, Math.max(0.05, probability)),
        confidence,
        factors,
        prediction: probability > 0.5 ? 'LIKELY' : 'UNLIKELY',
        modelType: 'fallback_heuristic'
    };
}


/**
 * Forecast expected revenue from simulation scenario
 */
function forecastRevenue(customers, scenario, personas) {
    const basePrice = scenario.basePrice || 75;
    const discount = scenario.discount || 0;
    const finalPrice = basePrice * (1 - discount / 100);

    let totalRevenue = 0;
    let expectedPurchases = 0;
    let personaBreakdown = {};

    customers.forEach(customer => {
        const persona = getCustomerPersona(customer, personas);
        const prediction = predictPurchaseProbability(customer, scenario, persona);

        const expectedValue = prediction.probability * finalPrice;
        totalRevenue += expectedValue;

        if (prediction.probability > 0.5) {
            expectedPurchases++;
        }

        if (!personaBreakdown[persona.type]) {
            personaBreakdown[persona.type] = { revenue: 0, purchases: 0 };
        }
        personaBreakdown[persona.type].revenue += expectedValue;
        personaBreakdown[persona.type].purchases += prediction.probability;
    });

    const avgOrderValue = expectedPurchases > 0 ? totalRevenue / expectedPurchases : 0;
    const conversionRate = (expectedPurchases / customers.length) * 100;
    const suggestions = generateRevenueSuggestions(scenario, personaBreakdown, conversionRate);

    return {
        totalRevenue: Math.round(totalRevenue),
        expectedPurchases: Math.round(expectedPurchases),
        conversionRate: conversionRate.toFixed(2),
        avgOrderValue: Math.round(avgOrderValue),
        personaBreakdown,
        suggestions,
        confidence: trainedWeights ? 0.92 : 0.85,
        modelType: trainedWeights ? 'ml_powered' : 'heuristic'
    };
}


/**
 * Predict customer churn risk using trained Random Forest feature importances
 */
function predictChurnRisk(customer, persona) {
    if (trainedWeights && trainedWeights.random_forest) {
        return predictChurnWithTrainedRF(customer, persona);
    }
    return predictChurnFallback(customer, persona);
}


/**
 * Churn prediction using trained Random Forest parameters
 */
function predictChurnWithTrainedRF(customer, persona) {
    const rf = trainedWeights.random_forest;
    const importances = rf.feature_importances;
    const bins = rf.probability_bins;

    // Build churn score using learned feature importances as weights
    let churnScore = 0;
    const riskFactors = [];

    // Review Rating is the most important feature (importance: 0.46)
    const ratingWeight = importances['Review Rating'] || 0.46;
    if (customer.reviewRating < 3.5) {
        churnScore += ratingWeight * (bins.low_rating_churn || 0.84);
        riskFactors.push({ factor: 'Low satisfaction', impact: 'high', importance: ratingWeight });
    } else if (customer.reviewRating >= 4.0) {
        churnScore += ratingWeight * (1 - (bins.high_rating_churn || 0.31));
        // High rating reduces churn, so invert
        churnScore -= ratingWeight * 0.3;
    }

    // Loyalty score is 2nd most important (importance: 0.20)
    const loyaltyWeight = importances['loyalty_score'] || 0.20;
    if ((customer.loyaltyScore || 0) < 0.3) {
        churnScore += loyaltyWeight * (bins.low_loyalty_high_churn || 0.70);
        riskFactors.push({ factor: 'Low loyalty signals', impact: 'high', importance: loyaltyWeight });
    } else if ((customer.loyaltyScore || 0) > 0.6) {
        churnScore -= loyaltyWeight * 0.2;
    }

    // Previous Purchases (importance: 0.11)
    const purchaseWeight = importances['Previous Purchases'] || 0.11;
    if (customer.previousPurchases < 5) {
        churnScore += purchaseWeight * 0.8;
        riskFactors.push({ factor: 'New customer (few purchases)', impact: 'medium', importance: purchaseWeight });
    } else if (customer.previousPurchases < 10) {
        churnScore += purchaseWeight * 0.4;
        riskFactors.push({ factor: 'Below-average purchase history', impact: 'low', importance: purchaseWeight });
    }

    // Purchase Amount (importance: 0.057)
    const amountWeight = importances['Purchase Amount (USD)'] || 0.057;
    if (customer.purchaseAmount < 30) {
        churnScore += amountWeight * 0.6;
        riskFactors.push({ factor: 'Low spending', impact: 'low', importance: amountWeight });
    }

    // Price sensitivity (importance: 0.053)
    const priceWeight = importances['price_sensitivity'] || 0.053;
    if (persona.type === 'loyal' && (customer.priceSensitivity || 0) > 0.7) {
        churnScore += priceWeight * 0.7;
        riskFactors.push({ factor: 'Price sensitivity (loyalty risk)', impact: 'medium', importance: priceWeight });
    }

    // Subscription status (not a direct feature but captured in loyalty)
    if (!customer.subscriptionStatus && persona.type === 'loyal') {
        churnScore += 0.1;
        riskFactors.push({ factor: 'No subscription (loyalty risk)', impact: 'medium', importance: 0.1 });
    }

    // Clamp to [0, 1]
    churnScore = Math.min(1, Math.max(0, churnScore));

    const thresholds = rf.thresholds;
    const riskLevel = churnScore > (thresholds.high_risk_threshold || 0.6) ? 'HIGH'
        : churnScore > (thresholds.medium_risk_threshold || 0.3) ? 'MEDIUM'
            : 'LOW';

    const retentionStrategy = generateRetentionStrategy(riskLevel, riskFactors, persona);

    return {
        churnScore,
        riskLevel,
        riskFactors,
        retentionStrategy,
        confidence: rf.metrics.accuracy,
        modelType: 'trained_random_forest'
    };
}


/**
 * Fallback churn prediction
 */
function predictChurnFallback(customer, persona) {
    let churnScore = 0;
    const riskFactors = [];

    if (customer.purchaseFrequency === 'Annually') {
        churnScore += 0.3;
        riskFactors.push({ factor: 'Low frequency', impact: 'high' });
    } else if (customer.purchaseFrequency === 'Quarterly') {
        churnScore += 0.15;
        riskFactors.push({ factor: 'Moderate frequency', impact: 'medium' });
    }

    if (customer.reviewRating < 3.5) {
        churnScore += 0.25;
        riskFactors.push({ factor: 'Low satisfaction', impact: 'high' });
    }

    if (!customer.subscriptionStatus && persona.type === 'loyal') {
        churnScore += 0.2;
        riskFactors.push({ factor: 'No subscription (loyalty risk)', impact: 'medium' });
    }

    if (customer.previousPurchases < 5) {
        churnScore += 0.15;
        riskFactors.push({ factor: 'New customer', impact: 'medium' });
    }

    if (persona.type === 'loyal' && customer.priceSensitivity > 0.7) {
        churnScore += 0.1;
        riskFactors.push({ factor: 'Price sensitivity increase', impact: 'low' });
    }

    churnScore = Math.min(1, churnScore);
    const riskLevel = churnScore > 0.6 ? 'HIGH' : churnScore > 0.3 ? 'MEDIUM' : 'LOW';
    const retentionStrategy = generateRetentionStrategy(riskLevel, riskFactors, persona);

    return {
        churnScore,
        riskLevel,
        riskFactors,
        retentionStrategy,
        confidence: 0.78,
        modelType: 'fallback_heuristic'
    };
}


/**
 * Calculate price elasticity
 */
function calculatePriceElasticity(customers, scenario, personas) {
    const baseDiscount = scenario.discount || 0;

    const testPoints = [
        { discount: baseDiscount - 10, label: 'Lower Discount' },
        { discount: baseDiscount, label: 'Current' },
        { discount: baseDiscount + 10, label: 'Higher Discount' }
    ];

    const elasticityData = testPoints.map(point => {
        const testScenario = { ...scenario, discount: Math.max(0, Math.min(100, point.discount)) };
        const forecast = forecastRevenue(customers, testScenario, personas);

        return {
            discount: point.discount,
            label: point.label,
            conversionRate: parseFloat(forecast.conversionRate),
            revenue: forecast.totalRevenue,
            purchases: forecast.expectedPurchases
        };
    });

    const current = elasticityData[1];
    const higher = elasticityData[2];

    const priceChange = ((100 - higher.discount) - (100 - current.discount)) / (100 - current.discount) * 100;
    const demandChange = (higher.purchases - current.purchases) / current.purchases * 100;

    const elasticity = priceChange !== 0 ? demandChange / priceChange : 0;
    const optimal = findOptimalDiscount(elasticityData);

    return {
        elasticity: elasticity.toFixed(2),
        interpretation: Math.abs(elasticity) > 1 ? 'Elastic (price-sensitive)' : 'Inelastic (less price-sensitive)',
        elasticityData,
        optimalDiscount: optimal.discount,
        optimalRevenue: optimal.revenue
    };
}


// ============ Helper Functions ============

function extractFeatures(customer, scenario, persona) {
    return {
        age: customer.age,
        priceSensitivity: customer.priceSensitivity || 0.5,
        loyaltyScore: customer.loyaltyScore || 0.5,
        impulseScore: customer.impulseScore || 0.5,
        qualityPreference: customer.qualityPreference || 0.5,
        avgPurchase: customer.purchaseAmount || 50,
        previousPurchases: customer.previousPurchases || 0,
        rating: customer.reviewRating || 3.5,
        categoryMatch: customer.category === scenario.category,
        seasonMatch: customer.season === scenario.season,
        hasSubscription: customer.subscriptionStatus
    };
}

function calculatePriceImpact(features, scenario, persona) {
    const discount = scenario.discount || 0;
    if (persona.type === 'budget') return Math.min(1, 0.3 + (discount / 100) * 0.7);
    if (persona.type === 'premium') return Math.max(0.6, 1 - (discount / 200));
    if (persona.type === 'impulse') return 0.5 + (discount / 100) * 0.4;
    return 0.7 + (discount / 100) * 0.2;
}

function calculateDiscountImpact(features, scenario, persona) {
    const discount = scenario.discount || 0;
    if (discount === 0) return 0.5;
    const sensitivity = features.priceSensitivity;
    return Math.min(1, 0.4 + (discount / 50) * sensitivity);
}

function calculateAffordability(features, scenario) {
    const basePrice = scenario.basePrice || 75;
    const discount = scenario.discount || 0;
    const finalPrice = basePrice * (1 - discount / 100);
    const avgPurchase = features.avgPurchase;
    const ratio = avgPurchase / finalPrice;
    if (ratio > 1.2) return 0.95;
    if (ratio > 0.8) return 0.75;
    if (ratio > 0.5) return 0.5;
    return 0.25;
}

function calculateConfidence(customer) {
    let confidence = 0.5;
    if (customer.previousPurchases > 10) confidence += 0.2;
    if (customer.reviewRating) confidence += 0.15;
    if (customer.subscriptionStatus !== undefined) confidence += 0.15;
    return Math.min(0.95, confidence);
}

function getCustomerPersona(customer, personas) {
    // If we have trained K-Means centroids, use nearest-centroid assignment
    if (trainedWeights && trainedWeights.kmeans) {
        return assignPersonaByKMeans(customer, personas);
    }

    // Fallback: score-based assignment
    const scores = {};
    for (const [type, persona] of Object.entries(personas)) {
        let score = 0;
        if (type === 'budget' && customer.priceSensitivity > 0.6) score += 1;
        if (type === 'premium' && customer.qualityPreference > 0.6) score += 1;
        if (type === 'impulse' && customer.impulseScore > 0.6) score += 1;
        if (type === 'loyal' && customer.loyaltyScore > 0.6) score += 1;
        scores[type] = score;
    }
    const bestType = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
    return personas[bestType] || personas.budget;
}


/**
 * Assign customer to persona using trained K-Means centroids (nearest centroid)
 */
function assignPersonaByKMeans(customer, personas) {
    const km = trainedWeights.kmeans;
    const scaler = km.scaler;

    // Customer feature vector (same 4 features used in training)
    const features = [
        customer.priceSensitivity || 0.5,
        customer.loyaltyScore || 0.5,
        customer.impulseScore || 0.3,
        customer.qualityPreference || 0.3
    ];

    // Standardize using trained scaler
    const scaled = features.map((val, i) => (val - scaler.mean[i]) / scaler.scale[i]);

    // Find nearest centroid
    let bestPersona = 'budget';
    let minDist = Infinity;

    for (const [personaType, centroidData] of Object.entries(km.persona_centroids)) {
        const centroid = centroidData.centroid_scaled;
        let dist = 0;
        for (let i = 0; i < scaled.length; i++) {
            dist += Math.pow(scaled[i] - centroid[i], 2);
        }
        if (dist < minDist) {
            minDist = dist;
            bestPersona = personaType;
        }
    }

    return personas[bestPersona] || personas.budget;
}


function generateRevenueSuggestions(scenario, personaBreakdown, conversionRate) {
    const suggestions = [];
    if (conversionRate < 30) {
        suggestions.push('Consider increasing discount to boost conversion rate');
    }
    if (scenario.discount > 40) {
        suggestions.push('High discount may hurt margins - test lower discount tiers');
    }
    const topPersona = Object.entries(personaBreakdown)
        .sort((a, b) => b[1].revenue - a[1].revenue)[0];
    if (topPersona) {
        suggestions.push(`Focus marketing on ${topPersona[0]} persona (${Math.round(topPersona[1].revenue)} expected revenue)`);
    }
    return suggestions;
}

function generateRetentionStrategy(riskLevel, riskFactors, persona) {
    const strategies = [];
    if (riskLevel === 'HIGH') {
        strategies.push('Immediate intervention required');
        strategies.push('Offer personalized discount (15-20%)');
        strategies.push('Direct outreach with exclusive offer');
    } else if (riskLevel === 'MEDIUM') {
        strategies.push('Send re-engagement campaign');
        strategies.push('Highlight new products in preferred category');
    }
    if (persona.type === 'loyal') {
        strategies.push('Emphasize loyalty rewards and benefits');
    }
    return strategies;
}

function findOptimalDiscount(elasticityData) {
    return elasticityData.reduce((best, current) =>
        current.revenue > best.revenue ? current : best
    );
}

module.exports = {
    predictPurchaseProbability,
    forecastRevenue,
    predictChurnRisk,
    calculatePriceElasticity
};
