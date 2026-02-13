/**
 * RESONANCE - AI-Powered Behavioral Simulation Server
 * 
 * This server powers the Digital Twin simulation by:
 * 1. Loading and analyzing customer data from CSV
 * 2. Creating AI persona agents using Grok/Gemini
 * 3. Streaming real-time thoughts via Server-Sent Events
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const { runSimulation, getPersonaInsights } = require('./src/agents/orchestrator');
const { loadAndAnalyzeData } = require('./src/data/csvProcessor');
const { initializeDatabase, saveSimulation, getSimulations, clearSimulations } = require('./src/data/db');
const { predictPurchaseProbability, forecastRevenue, predictChurnRisk, calculatePriceElasticity } = require('./src/ml/predictiveModels');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// Store analyzed data in memory
let customerData = null;
let personaProfiles = null;

// Initialize data on startup
async function initializeData() {
    try {
        console.log('🔄 Loading customer data...');
        const result = await loadAndAnalyzeData(path.join(__dirname, 'src', 'ml', 'shopping_trends.csv'));
        customerData = result.customers;
        personaProfiles = result.personas;
        console.log(`✅ Loaded ${customerData.length} customers`);
        console.log(`✅ Created ${Object.keys(personaProfiles).length} persona profiles`);
    } catch (error) {
        console.error('❌ Error loading data:', error.message);
    }
}

// API Routes

/**
 * GET /api/personas
 * Returns the discovered persona profiles from the data
 */
app.get('/api/personas', (req, res) => {
    if (!personaProfiles) {
        return res.status(503).json({ error: 'Data not loaded yet' });
    }
    res.json(personaProfiles);
});

/**
 * GET /api/stats
 * Returns overall statistics about the customer data
 */
app.get('/api/stats', (req, res) => {
    if (!customerData) {
        return res.status(503).json({ error: 'Data not loaded yet' });
    }

    res.json({
        totalCustomers: customerData.length,
        personaCount: Object.keys(personaProfiles).length,
        categories: [...new Set(customerData.map(c => c.category))],
        avgPurchaseAmount: Math.round(customerData.reduce((sum, c) => sum + c.purchaseAmount, 0) / customerData.length),
        avgRating: (customerData.reduce((sum, c) => sum + c.reviewRating, 0) / customerData.length).toFixed(1)
    });
});

/**
 * POST /api/simulate
 * Run a simulation with given parameters
 * Returns aggregated results (non-streaming)
 */
app.post('/api/simulate', async (req, res) => {
    try {
        const { discount, category, brand, season } = req.body;

        if (!personaProfiles) {
            return res.status(503).json({ error: 'Data not loaded yet' });
        }

        const results = await runSimulation({
            discount: discount || 20,
            category: category || 'Clothing',
            brand: brand || 'premium',
            season: season || 'Spring',
            personas: personaProfiles,
            customers: customerData  // Pass for ML predictions
        });

        res.json(results);
    } catch (error) {
        console.error('Simulation error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/simulate/stream
 * Stream simulation thoughts in real-time via SSE
 */
app.get('/api/simulate/stream', async (req, res) => {
    // Set up SSE headers
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');

    const { discount, category, brand, season } = req.query;

    if (!personaProfiles) {
        res.write(`data: ${JSON.stringify({ type: 'error', message: 'Data not loaded yet' })}\n\n`);
        res.end();
        return;
    }

    try {
        // Send start event
        res.write(`data: ${JSON.stringify({ type: 'start', message: 'Simulation starting...' })}\n\n`);

        // Run simulation with streaming callback
        await runSimulation({
            discount: parseInt(discount) || 20,
            category: category || 'Clothing',
            brand: brand || 'premium',
            season: season || 'Spring',
            personas: personaProfiles,
            onThought: (thought) => {
                res.write(`data: ${JSON.stringify({ type: 'thought', ...thought })}\n\n`);
            },
            onMetrics: (metrics) => {
                res.write(`data: ${JSON.stringify({ type: 'metrics', ...metrics })}\n\n`);
            },
            onProgress: (progress) => {
                res.write(`data: ${JSON.stringify({ type: 'progress', progress })}\n\n`);
            }
        });

        // Send complete event
        res.write(`data: ${JSON.stringify({ type: 'complete', message: 'Simulation complete' })}\n\n`);
        res.end();

    } catch (error) {
        console.error('Stream error:', error);
        res.write(`data: ${JSON.stringify({ type: 'error', message: error.message })}\n\n`);
        res.end();
    }
});

/**
 * GET /api/persona/:type/insights
 * Get detailed insights for a specific persona type
 */
app.get('/api/persona/:type/insights', async (req, res) => {
    try {
        const { type } = req.params;
        const { category, season } = req.query;

        if (!personaProfiles || !personaProfiles[type]) {
            return res.status(404).json({ error: 'Persona not found' });
        }

        const insights = await getPersonaInsights(personaProfiles[type], {
            category: category || 'Clothing',
            season: season || 'Spring'
        });

        res.json(insights);
    } catch (error) {
        console.error('Insights error:', error);
        res.status(500).json({ error: error.message });
    }
});

// ===== ML PREDICTION ENDPOINTS =====

/**
 * POST /api/ml/forecast
 * Forecast revenue using ML models
 */
app.post('/api/ml/forecast', async (req, res) => {
    try {
        const { discount, category, brand, season, basePrice } = req.body;

        if (!customerData || !personaProfiles) {
            return res.status(503).json({ error: 'Data not loaded yet' });
        }

        const forecast = forecastRevenue(customerData, {
            discount: parseFloat(discount) || 20,
            category: category || 'Clothing',
            brand: brand || 'premium',
            season: season || 'Spring',
            basePrice: parseFloat(basePrice) || 75
        }, personaProfiles);

        res.json(forecast);
    } catch (error) {
        console.error('ML Forecast error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/ml/elasticity
 * Calculate price elasticity
 */
app.post('/api/ml/elasticity', async (req, res) => {
    try {
        const { discount, category, brand, season } = req.body;

        if (!customerData || !personaProfiles) {
            return res.status(503).json({ error: 'Data not loaded yet' });
        }

        const elasticity = calculatePriceElasticity(customerData, {
            discount: parseFloat(discount) || 20,
            category: category || 'Clothing',
            brand: brand || 'premium',
            season: season || 'Spring'
        }, personaProfiles);

        res.json(elasticity);
    } catch (error) {
        console.error('ML Elasticity error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/ml/churn
 * Predict churn risk for customers
 */
app.post('/api/ml/churn', async (req, res) => {
    try {
        const { customerId } = req.body;

        if (!customerData || !personaProfiles) {
            return res.status(503).json({ error: 'Data not loaded yet' });
        }

        // Get sample of high-risk customers
        const churnPredictions = customerData.slice(0, 50).map(customer => {
            const persona = Object.values(personaProfiles)[0]; // Simplified
            const prediction = predictChurnRisk(customer, persona);

            return {
                customerId: customer.id,
                ...prediction
            };
        }).filter(p => p.riskLevel !== 'LOW').sort((a, b) => b.churnScore - a.churnScore);

        res.json({ predictions: churnPredictions.slice(0, 10) });
    } catch (error) {
        console.error('ML Churn error:', error);
        res.status(500).json({ error: error.message });
    }
});

// ===== SIMULATION HISTORY API =====

/**
 * GET /api/simulations
 * Load all saved simulations from database
 */
app.get('/api/simulations', async (req, res) => {
    try {
        const simulations = await getSimulations();
        res.json(simulations);
    } catch (error) {
        console.error('Error loading simulations:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/simulations
 * Save a new simulation to database
 */
app.post('/api/simulations', async (req, res) => {
    try {
        const simulation = req.body;
        const saved = await saveSimulation(simulation);
        res.json({ success: true, simulation: saved });
    } catch (error) {
        console.error('Error saving simulation:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * DELETE /api/simulations
 * Clear all simulation history
 */
app.delete('/api/simulations', async (req, res) => {
    try {
        await clearSimulations();
        res.json({ success: true });
    } catch (error) {
        console.error('Error clearing simulations:', error);
        res.status(500).json({ error: error.message });
    }
});

// Health check
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        dataLoaded: !!customerData,
        personasReady: !!personaProfiles,
        aiProvider: process.env.AI_PROVIDER || 'gemini',
        databaseConnected: !!process.env.SUPABASE_URL
    });
});

// Favicon handler
app.get('/favicon.ico', (req, res) => res.sendFile(path.join(__dirname, 'favicon.svg')));

// Serve main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start server
app.listen(PORT, async () => {
    console.log(`
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🔮 RESONANCE - Behavioral Simulation Platform           ║
║                                                           ║
║   Server running at: http://localhost:${PORT}               ║
║   AI Provider: ${(process.env.AI_PROVIDER || 'gemini').padEnd(40)}║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    `);

    // Initialize database
    initializeDatabase();

    // Initialize data
    await initializeData();
});

module.exports = app;
