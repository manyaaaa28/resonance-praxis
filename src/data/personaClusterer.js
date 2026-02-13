/**
 * Persona Clusterer for RESONANCE
 * 
 * Analyzes customer data to identify behavioral segments
 * and creates rich persona profiles for AI agents.
 * 
 * Uses trained K-Means centroids (from train_models.py) for
 * real ML-based clustering when available, with fallback to
 * threshold-based segmentation.
 */

const fs = require('fs');
const path = require('path');

// Load trained K-Means model weights
let kmeansModel = null;

try {
    const weightsPath = path.join(__dirname, '..', 'ml', 'trained_model_weights.json');
    const rawData = fs.readFileSync(weightsPath, 'utf-8');
    const trainedWeights = JSON.parse(rawData);
    kmeansModel = trainedWeights.kmeans;
    console.log(`[Clusterer] Loaded trained K-Means model (k=${kmeansModel.k}, silhouette=${kmeansModel.metrics.silhouette_score})`);
} catch (err) {
    console.warn('[Clusterer] No trained K-Means model found - using threshold-based segmentation');
}


/**
 * Assign a single customer to a persona using K-Means nearest centroid
 */
function assignToNearestCentroid(customer) {
    if (!kmeansModel) return null;

    const scaler = kmeansModel.scaler;
    const features = [
        customer.priceSensitivity,
        customer.loyaltyScore,
        customer.impulseScore,
        customer.qualityPreference
    ];

    // Standardize using trained scaler parameters
    const scaled = features.map((val, i) => (val - scaler.mean[i]) / scaler.scale[i]);

    // Find nearest centroid (Euclidean distance)
    let bestPersona = 'budget';
    let minDist = Infinity;

    for (const [personaType, centroidData] of Object.entries(kmeansModel.persona_centroids)) {
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

    return bestPersona;
}


/**
 * Cluster customers into behavioral personas
 * @param {Array} customers - Array of customer objects
 * @returns {Object} - Persona profiles keyed by type
 */
function clusterCustomers(customers) {
    // Calculate customer metrics for clustering
    const enrichedCustomers = customers.map(c => ({
        ...c,
        // Price sensitivity score (0-1, higher = more price sensitive)
        priceSensitivity: calculatePriceSensitivity(c),
        // Loyalty score (0-1, higher = more loyal)
        loyaltyScore: calculateLoyaltyScore(c),
        // Impulse score (0-1, higher = more impulsive)
        impulseScore: calculateImpulseScore(c),
        // Quality preference (0-1, higher = prefers premium)
        qualityPreference: calculateQualityPreference(c)
    }));

    // Use trained K-Means centroids if available (real ML clustering)
    if (kmeansModel) {
        console.log('[Clusterer] Using trained K-Means centroids for segmentation');

        // Assign each customer to their nearest centroid
        const segments = { budget: [], premium: [], impulse: [], loyal: [] };

        enrichedCustomers.forEach(c => {
            const persona = assignToNearestCentroid(c);
            if (segments[persona]) {
                segments[persona].push(c);
            }
        });

        const personas = {
            budget: createPersonaProfile('budget', segments.budget),
            premium: createPersonaProfile('premium', segments.premium),
            impulse: createPersonaProfile('impulse', segments.impulse),
            loyal: createPersonaProfile('loyal', segments.loyal)
        };

        console.log(`[Clusterer] Segmentation: budget=${segments.budget.length}, premium=${segments.premium.length}, impulse=${segments.impulse.length}, loyal=${segments.loyal.length}`);
        return personas;
    }

    // Fallback: threshold-based segmentation
    console.log('[Clusterer] Using threshold-based segmentation (no trained model)');
    const personas = {
        budget: createPersonaProfile('budget', enrichedCustomers.filter(c => c.priceSensitivity > 0.6)),
        premium: createPersonaProfile('premium', enrichedCustomers.filter(c => c.qualityPreference > 0.6)),
        impulse: createPersonaProfile('impulse', enrichedCustomers.filter(c => c.impulseScore > 0.6)),
        loyal: createPersonaProfile('loyal', enrichedCustomers.filter(c => c.loyaltyScore > 0.6))
    };

    return personas;
}

/**
 * Calculate price sensitivity based on behavior
 */
function calculatePriceSensitivity(customer) {
    let score = 0;

    // Lower purchase amounts indicate price sensitivity
    if (customer.purchaseAmount < 40) score += 0.3;
    else if (customer.purchaseAmount < 60) score += 0.15;

    // Always uses discounts/promos
    if (customer.discountApplied) score += 0.2;
    if (customer.promoCodeUsed) score += 0.2;

    // Chooses free/standard shipping
    if (customer.shippingType === 'Free Shipping') score += 0.15;
    if (customer.shippingType === 'Standard') score += 0.1;

    // Payment method preferences
    if (customer.preferredPaymentMethod === 'Cash') score += 0.05;

    return Math.min(1, score);
}

/**
 * Calculate loyalty score based on behavior
 */
function calculateLoyaltyScore(customer) {
    let score = 0;

    // High previous purchases
    if (customer.previousPurchases > 40) score += 0.35;
    else if (customer.previousPurchases > 25) score += 0.2;
    else if (customer.previousPurchases > 10) score += 0.1;

    // Has subscription
    if (customer.subscriptionStatus) score += 0.25;

    // High ratings indicate satisfaction
    if (customer.reviewRating >= 4.5) score += 0.2;
    else if (customer.reviewRating >= 4.0) score += 0.1;

    // Frequent purchases
    if (['Weekly', 'Bi-Weekly', 'Fortnightly'].includes(customer.purchaseFrequency)) {
        score += 0.2;
    }

    return Math.min(1, score);
}

/**
 * Calculate impulse buying tendency
 */
function calculateImpulseScore(customer) {
    let score = 0;

    // Express/Next Day shipping suggests urgency
    if (customer.shippingType === 'Next Day Air') score += 0.3;
    if (customer.shippingType === 'Express') score += 0.2;

    // Younger customers tend to be more impulsive
    if (customer.age < 30) score += 0.2;
    else if (customer.age < 40) score += 0.1;

    // Quick payment methods
    if (['Venmo', 'PayPal'].includes(customer.paymentMethod)) score += 0.15;

    // Accessories often impulse buys
    if (customer.category === 'Accessories') score += 0.15;

    // Moderate ratings (not overthinking)
    if (customer.reviewRating >= 3.5 && customer.reviewRating <= 4.2) score += 0.1;

    return Math.min(1, score);
}

/**
 * Calculate quality/premium preference
 */
function calculateQualityPreference(customer) {
    let score = 0;

    // Higher purchase amounts
    if (customer.purchaseAmount > 80) score += 0.3;
    else if (customer.purchaseAmount > 60) score += 0.15;

    // Premium shipping choices
    if (customer.shippingType === 'Next Day Air') score += 0.15;
    if (customer.shippingType === '2-Day Shipping') score += 0.1;

    // Credit card suggests established credit
    if (customer.paymentMethod === 'Credit Card') score += 0.1;
    if (customer.preferredPaymentMethod === 'Credit Card') score += 0.1;

    // High ratings = quality conscious
    if (customer.reviewRating >= 4.5) score += 0.15;

    // Older customers often prefer quality
    if (customer.age > 45) score += 0.1;

    // Doesn't always chase discounts
    if (!customer.discountApplied) score += 0.1;

    return Math.min(1, score);
}

/**
 * Create a rich persona profile from customer segment
 */
function createPersonaProfile(type, customers) {
    if (customers.length === 0) {
        return getDefaultProfile(type);
    }

    const avgAge = Math.round(customers.reduce((sum, c) => sum + c.age, 0) / customers.length);
    const avgPurchase = Math.round(customers.reduce((sum, c) => sum + c.purchaseAmount, 0) / customers.length);
    const avgRating = (customers.reduce((sum, c) => sum + c.reviewRating, 0) / customers.length).toFixed(1);
    const avgPrevPurchases = Math.round(customers.reduce((sum, c) => sum + c.previousPurchases, 0) / customers.length);

    // Most common values
    const topLocation = getMostCommon(customers.map(c => c.location));
    const topCategory = getMostCommon(customers.map(c => c.category));
    const topPayment = getMostCommon(customers.map(c => c.preferredPaymentMethod));
    const topFrequency = getMostCommon(customers.map(c => c.purchaseFrequency));
    const subscriptionRate = Math.round((customers.filter(c => c.subscriptionStatus).length / customers.length) * 100);

    return {
        type,
        count: customers.length,
        demographics: {
            avgAge,
            topLocation,
            genderSplit: getGenderSplit(customers)
        },
        behavior: {
            avgPurchaseAmount: avgPurchase,
            avgRating: parseFloat(avgRating),
            avgPreviousPurchases: avgPrevPurchases,
            topCategory,
            preferredPayment: topPayment,
            purchaseFrequency: topFrequency,
            subscriptionRate
        },
        traits: getPersonaTraits(type, { avgPurchase, avgAge, subscriptionRate, topPayment }),
        prompt: generateSystemPrompt(type, { avgAge, avgPurchase, topLocation, subscriptionRate, topCategory, topPayment })
    };
}

/**
 * Get most common value in array
 */
function getMostCommon(arr) {
    const counts = {};
    arr.forEach(val => { counts[val] = (counts[val] || 0) + 1; });
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'Unknown';
}

/**
 * Get gender split
 */
function getGenderSplit(customers) {
    const male = customers.filter(c => c.gender === 'Male').length;
    const female = customers.filter(c => c.gender === 'Female').length;
    return { male, female };
}

/**
 * Get persona traits for display
 */
function getPersonaTraits(type, stats) {
    const traits = {
        budget: ['Price Sensitive', 'Deal Seeker', 'Compares Options', 'Values Savings'],
        premium: ['Quality Focused', 'Brand Conscious', 'Willing to Pay More', 'Expects Excellence'],
        impulse: ['Spontaneous', 'Trend Follower', 'Quick Decisions', 'Easily Excited'],
        loyal: ['Repeat Buyer', 'Brand Advocate', 'Value Relationships', 'Trusts Recommendations']
    };
    return traits[type] || ['Customer'];
}

/**
 * Generate system prompt for AI agent
 */
function generateSystemPrompt(type, stats) {
    const prompts = {
        budget: `You are a Budget Hunter customer persona. You are ${stats.avgAge} years old, typically from ${stats.topLocation}. 
You are extremely price-sensitive and always looking for the best deals. Your average purchase is around $${stats.avgPurchase}.
You compare prices, wait for sales, use promo codes, and prefer ${stats.topPayment} for payments.
${stats.subscriptionRate > 50 ? 'You have a subscription for extra savings.' : 'You avoid subscriptions to stay flexible.'}
You love ${stats.topCategory} products but only buy when the price is right.
When evaluating a purchase, you always consider: Is this the best price? Can I find it cheaper? Is there a coupon?`,

        premium: `You are a Premium Buyer customer persona. You are ${stats.avgAge} years old, often from ${stats.topLocation}.
You prioritize quality over price and are willing to pay more for excellence. Your average purchase is $${stats.avgPurchase}.
You prefer ${stats.topPayment} for payments and expect premium service.
${stats.subscriptionRate > 50 ? 'You subscribe for exclusive member benefits.' : 'You prefer to shop premium items individually.'}
You love high-quality ${stats.topCategory} products.
When evaluating a purchase, you consider: Is this the best quality? Does this brand have good reputation? Will this last?`,

        impulse: `You are an Impulse Shopper customer persona. You are ${stats.avgAge} years old, often from ${stats.topLocation}.
You make quick purchase decisions based on emotion and trends. Your average purchase is $${stats.avgPurchase}.
You prefer fast payment methods like ${stats.topPayment} and love instant gratification.
${stats.subscriptionRate > 50 ? 'You enjoy subscription surprises.' : 'You prefer spontaneous shopping over subscriptions.'}
You're drawn to trendy ${stats.topCategory} items.
When evaluating a purchase, you think: This looks amazing! I want it now! It's calling my name!`,

        loyal: `You are a Loyal Customer persona. You are ${stats.avgAge} years old, typically from ${stats.topLocation}.
You value long-term relationships with brands and keep coming back. Your average purchase is $${stats.avgPurchase}.
You prefer ${stats.topPayment} and appreciate loyalty rewards.
${stats.subscriptionRate > 50 ? 'You maintain subscriptions for reliability and perks.' : 'You show loyalty through repeat purchases.'}
You regularly buy ${stats.topCategory} from your trusted brands.
When evaluating a purchase, you consider: Have I bought from them before? Do they reward my loyalty? Can I recommend this?`
    };

    return prompts[type] || 'You are a customer evaluating a purchase.';
}

/**
 * Default profile if segment is empty
 */
function getDefaultProfile(type) {
    const defaults = {
        budget: { avgAge: 32, avgPurchase: 35, topLocation: 'California', subscriptionRate: 40, topCategory: 'Clothing', topPayment: 'Cash' },
        premium: { avgAge: 45, avgPurchase: 85, topLocation: 'New York', subscriptionRate: 70, topCategory: 'Clothing', topPayment: 'Credit Card' },
        impulse: { avgAge: 28, avgPurchase: 55, topLocation: 'Texas', subscriptionRate: 50, topCategory: 'Accessories', topPayment: 'Venmo' },
        loyal: { avgAge: 42, avgPurchase: 65, topLocation: 'Florida', subscriptionRate: 80, topCategory: 'Clothing', topPayment: 'Credit Card' }
    };

    const d = defaults[type];
    return {
        type,
        count: 0,
        demographics: { avgAge: d.avgAge, topLocation: d.topLocation, genderSplit: { male: 50, female: 50 } },
        behavior: { avgPurchaseAmount: d.avgPurchase, avgRating: 4.0, avgPreviousPurchases: 20, topCategory: d.topCategory, preferredPayment: d.topPayment, purchaseFrequency: 'Monthly', subscriptionRate: d.subscriptionRate },
        traits: getPersonaTraits(type, d),
        prompt: generateSystemPrompt(type, d)
    };
}

module.exports = {
    clusterCustomers,
    calculatePriceSensitivity,
    calculateLoyaltyScore,
    calculateImpulseScore,
    calculateQualityPreference
};
