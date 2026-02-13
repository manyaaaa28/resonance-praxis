/**
 * Supabase Database Module for RESONANCE
 * Handles simulation history persistence
 */

const { createClient } = require('@supabase/supabase-js');

// Initialize Supabase client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_ANON_KEY;

let supabase = null;

function initializeDatabase() {
    if (!supabaseUrl || !supabaseKey) {
        console.warn('⚠️ Supabase credentials not found - database disabled');
        return false;
    }

    supabase = createClient(supabaseUrl, supabaseKey);
    console.log('✅ Supabase database connected');
    return true;
}

/**
 * Save a simulation to the database
 * @param {Object} simulation - Simulation data
 * @returns {Object} - Saved simulation with ID
 */
async function saveSimulation(simulation) {
    if (!supabase) {
        console.warn('Database not initialized');
        return null;
    }

    const { data, error } = await supabase
        .from('simulations')
        .insert([{
            category: simulation.category,
            brand: simulation.brand,
            discount: simulation.discount,
            season: simulation.season,
            revenue_change: simulation.results?.revenueChange || simulation.revenueChange,
            conversion: simulation.results?.conversion || simulation.conversion,
            churn: simulation.results?.churn || simulation.churn
        }])
        .select();

    if (error) {
        console.error('Error saving simulation:', error.message);
        return null;
    }

    return data[0];
}

/**
 * Get all simulations from the database
 * @returns {Array} - Array of simulations
 */
async function getSimulations() {
    if (!supabase) {
        return [];
    }

    const { data, error } = await supabase
        .from('simulations')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(50);

    if (error) {
        console.error('Error loading simulations:', error.message);
        return [];
    }

    // Transform to frontend format
    return data.map(sim => ({
        id: sim.id,
        category: sim.category,
        brand: sim.brand,
        discount: sim.discount,
        season: sim.season,
        results: {
            revenueChange: sim.revenue_change,
            conversion: sim.conversion,
            churn: sim.churn
        },
        createdAt: sim.created_at
    }));
}

/**
 * Clear all simulations from the database
 */
async function clearSimulations() {
    if (!supabase) {
        return false;
    }

    const { error } = await supabase
        .from('simulations')
        .delete()
        .neq('id', 0); // Delete all rows

    if (error) {
        console.error('Error clearing simulations:', error.message);
        return false;
    }

    return true;
}

module.exports = {
    initializeDatabase,
    saveSimulation,
    getSimulations,
    clearSimulations
};
