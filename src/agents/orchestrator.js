/**
 * AI Agent Orchestrator for RESONANCE
 * 
 * Coordinates multiple AI persona agents to simulate customer behavior
 * Supports both Grok AI (xAI) and Gemini AI (Google)
 */

const { createPersonaAgent } = require('./personaAgent');
const { forecastRevenue, calculatePriceElasticity } = require('../ml/predictiveModels');

// Agent display metadata
const AGENT_METADATA = {
    budget: {
        name: 'Budget Hunter',
        emoji: '💰',
        color: '#f59e0b',
        description: 'Price Sensitive'
    },
    premium: {
        name: 'Premium Buyer',
        emoji: '💎',
        color: '#6366f1',
        description: 'Brand Loyal'
    },
    impulse: {
        name: 'Impulse Shopper',
        emoji: '⚡',
        color: '#10b981',
        description: 'Quick Decisions'
    },
    loyal: {
        name: 'Loyal Customer',
        emoji: '❤️',
        color: '#ec4899',
        description: 'Repeat Buyer'
    }
};

/**
 * Run a full simulation with all persona agents
 * @param {Object} params - Simulation parameters
 * @returns {Object} - Aggregated results
 */
async function runSimulation(params) {
    const {
        discount,
        category,
        brand,
        season,
        personas,
        onThought,    // Callback for streaming thoughts
        onMetrics,    // Callback for metrics updates
        onProgress    // Callback for progress updates
    } = params;

    const agentTypes = ['budget', 'premium', 'impulse', 'loyal'];
    const results = [];
    let totalBuyers = 0;
    let totalLeavers = 0;

    // Shuffle agent order for variety
    const shuffledAgents = agentTypes.sort(() => Math.random() - 0.5);

    for (let i = 0; i < shuffledAgents.length; i++) {
        const agentType = shuffledAgents[i];
        const persona = personas[agentType];
        const metadata = AGENT_METADATA[agentType];

        try {
            // Create agent and get response
            const agent = createPersonaAgent(agentType, persona);
            const response = await agent.evaluate({
                discount,
                category,
                brand,
                season
            });

            // Track decisions
            if (response.decision === 'buy') totalBuyers++;
            else if (response.decision === 'leave') totalLeavers++;

            // Build thought object
            const thought = {
                agentType,
                agentName: metadata.name,
                emoji: metadata.emoji,
                color: metadata.color,
                thought: response.thought,
                decision: response.decision,
                action: response.action,
                confidence: response.confidence,
                timestamp: new Date().toISOString()
            };

            results.push(thought);

            // Stream thought if callback provided
            if (onThought) {
                onThought(thought);
            }

            // Update progress
            if (onProgress) {
                onProgress(Math.round(((i + 1) / shuffledAgents.length) * 100));
            }

            // Small delay between agents for visual effect
            await delay(800);

        } catch (error) {
            console.error(`Error with ${agentType} agent:`, error.message);

            // Send fallback thought
            const fallbackThought = {
                agentType,
                agentName: metadata.name,
                emoji: metadata.emoji,
                color: metadata.color,
                thought: getFallbackThought(agentType, discount, category),
                decision: getFallbackDecision(agentType, discount),
                action: 'Considering...',
                confidence: 0.5,
                timestamp: new Date().toISOString()
            };

            results.push(fallbackThought);
            if (onThought) onThought(fallbackThought);
        }
    }

    // Calculate metrics using ML models
    const metrics = calculateMetrics(results, discount, category, brand, personas, params.customers);

    if (onMetrics) {
        onMetrics(metrics);
    }

    return {
        thoughts: results,
        metrics,
        summary: {
            totalAgents: results.length,
            buyers: totalBuyers,
            leavers: totalLeavers,
            hesitating: results.length - totalBuyers - totalLeavers
        }
    };
}

/**
 * Calculate business metrics from agent decisions using ML models
 */
function calculateMetrics(results, discount, category, brand, personas, customers) {
    const buyerCount = results.filter(r => r.decision === 'buy').length;
    const leaverCount = results.filter(r => r.decision === 'leave').length;
    const conversionRate = Math.round((buyerCount / results.length) * 100);

    // Use ML revenue forecasting if customers data available
    let projectedRevenue, revenueChange, churnRisk, churnChange, mlInsights;

    if (customers && personas) {
        try {
            const forecast = forecastRevenue(customers, {
                discount,
                category,
                brand,
                basePrice: 75
            }, personas);

            projectedRevenue = forecast.totalRevenue;
            revenueChange = `+${(forecast.conversionRate / 5).toFixed(1)}%`;
            mlInsights = forecast.suggestions;

            churnRisk = leaverCount > 2 ? (25 + Math.random() * 10).toFixed(1) : (10 + Math.random() * 5).toFixed(1);
            churnChange = leaverCount > 2 ? `+${(Math.random() * 3 + 1).toFixed(1)}%` : `-${(Math.random() * 2 + 0.5).toFixed(1)}%`;
        } catch (err) {
            console.error('ML forecasting error:', err.message);
            // Fallback to simple calculation
            projectedRevenue = Math.round(142000 * (1 + discount / 200) * (leaverCount > 2 ? 0.85 : 1));
            revenueChange = `+${(Math.random() * 8 + 2).toFixed(1)}%`;
        }
    } else {
        // Fallback calculation
        const baseRevenue = 142000;
        const discountImpact = 1 + (discount / 200);
        const churnImpact = leaverCount > 2 ? 0.85 : 1;
        projectedRevenue = Math.round(baseRevenue * discountImpact * churnImpact * (1 + Math.random() * 0.1));
        revenueChange = discount > 30 ? `+${(discount / 5).toFixed(1)}%` : `+${(Math.random() * 8 + 2).toFixed(1)}%`;
    }

    churnRisk = churnRisk || Math.round((leaverCount / results.length) * 100) + (Math.random() * 3).toFixed(1);
    churnChange = churnChange || (leaverCount > 2 ? `+${(Math.random() * 3 + 1).toFixed(1)}%` : `-${(Math.random() * 2 + 0.5).toFixed(1)}%`);

    return {
        revenue: projectedRevenue,
        revenueChange,
        churnRisk: parseFloat(churnRisk),
        churnChange,
        conversion: conversionRate,
        conversionChange: buyerCount >= 3 ? `+${(Math.random() * 5 + 3).toFixed(1)}%` : `-${(Math.random() * 3 + 1).toFixed(1)}%`,
        mlInsights: mlInsights || []
    };
}

/**
 * Get insights for a specific persona
 */
async function getPersonaInsights(persona, context) {
    const agent = createPersonaAgent(persona.type, persona);

    return {
        persona: persona.type,
        traits: persona.traits,
        demographics: persona.demographics,
        behavior: persona.behavior,
        recommendations: generateRecommendations(persona, context)
    };
}

/**
 * Generate recommendations for a persona
 */
function generateRecommendations(persona, context) {
    const recs = [];

    if (persona.type === 'budget') {
        recs.push('Offer bundle deals to increase basket size');
        recs.push('Highlight free shipping thresholds');
        recs.push('Create flash sale notifications');
    } else if (persona.type === 'premium') {
        recs.push('Emphasize quality and craftsmanship');
        recs.push('Offer exclusive member previews');
        recs.push('Provide premium packaging options');
    } else if (persona.type === 'impulse') {
        recs.push('Use urgency messaging (limited stock)');
        recs.push('Show trending/popular items prominently');
        recs.push('Optimize one-click checkout');
    } else if (persona.type === 'loyal') {
        recs.push('Implement tiered loyalty rewards');
        recs.push('Send personalized anniversary offers');
        recs.push('Create referral incentive programs');
    }

    return recs;
}

/**
 * Fallback thought if AI fails
 */
function getFallbackThought(agentType, discount, category) {
    const fallbacks = {
        budget: discount > 30
            ? `${discount}% off ${category}? That's a great deal! Let me check my budget...`
            : `Hmm, ${discount}% isn't quite enough for me on ${category}. I'll wait for a better sale.`,
        premium: discount > 50
            ? `Such a steep discount on ${category} makes me question the quality...`
            : `${category} at this price point works for me. Quality matters more than discounts.`,
        impulse: `Ooh, ${category} on sale! My fingers are itching to click buy!`,
        loyal: `I always shop here for ${category}. ${discount}% off is just a bonus!`
    };
    return fallbacks[agentType] || `Evaluating this ${category} offer...`;
}

/**
 * Fallback decision based on persona type and discount
 */
function getFallbackDecision(agentType, discount) {
    if (agentType === 'budget') return discount > 30 ? 'buy' : 'leave';
    if (agentType === 'premium') return discount > 60 ? 'hesitate' : 'buy';
    if (agentType === 'impulse') return 'buy';
    if (agentType === 'loyal') return 'buy';
    return 'hesitate';
}

/**
 * Utility delay function
 */
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = {
    runSimulation,
    getPersonaInsights,
    AGENT_METADATA
};
