/**
 * CSV Data Processor for RESONANCE
 * 
 * Loads customer data from CSV and creates persona clusters
 * based on shopping behavior patterns
 */

const fs = require('fs');
const { parse } = require('csv-parse/sync');
const { clusterCustomers } = require('./personaClusterer');

/**
 * Load and analyze customer data from CSV
 * @param {string} filePath - Path to the CSV file
 * @returns {Object} - { customers, personas }
 */
async function loadAndAnalyzeData(filePath) {
    // Read and parse CSV
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const records = parse(fileContent, {
        columns: true,
        skip_empty_lines: true,
        trim: true
    });

    // Transform to clean customer objects
    const customers = records.map(row => ({
        id: parseInt(row['Customer ID']),
        age: parseInt(row['Age']),
        gender: row['Gender'],
        itemPurchased: row['Item Purchased'],
        category: row['Category'],
        purchaseAmount: parseFloat(row['Purchase Amount (USD)']),
        location: row['Location'],
        size: row['Size'],
        color: row['Color'],
        season: row['Season'],
        reviewRating: parseFloat(row['Review Rating']),
        subscriptionStatus: row['Subscription Status'] === 'Yes',
        paymentMethod: row['Payment Method'],
        shippingType: row['Shipping Type'],
        discountApplied: row['Discount Applied'] === 'Yes',
        promoCodeUsed: row['Promo Code Used'] === 'Yes',
        previousPurchases: parseInt(row['Previous Purchases']),
        preferredPaymentMethod: row['Preferred Payment Method'],
        purchaseFrequency: row['Frequency of Purchases']
    }));

    // Cluster customers into personas
    const personas = clusterCustomers(customers);

    return { customers, personas };
}

/**
 * Get statistics for a specific category
 */
function getCategoryStats(customers, category) {
    const filtered = customers.filter(c => c.category === category);
    if (filtered.length === 0) return null;

    return {
        count: filtered.length,
        avgAmount: Math.round(filtered.reduce((sum, c) => sum + c.purchaseAmount, 0) / filtered.length),
        avgRating: (filtered.reduce((sum, c) => sum + c.reviewRating, 0) / filtered.length).toFixed(1),
        topItems: getTopItems(filtered),
        seasonalTrends: getSeasonalTrends(filtered)
    };
}

/**
 * Get top purchased items
 */
function getTopItems(customers, limit = 5) {
    const itemCounts = {};
    customers.forEach(c => {
        itemCounts[c.itemPurchased] = (itemCounts[c.itemPurchased] || 0) + 1;
    });

    return Object.entries(itemCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, limit)
        .map(([item, count]) => ({ item, count }));
}

/**
 * Get seasonal purchase trends
 */
function getSeasonalTrends(customers) {
    const seasons = ['Spring', 'Summer', 'Fall', 'Winter'];
    const trends = {};

    seasons.forEach(season => {
        const seasonCustomers = customers.filter(c => c.season === season);
        trends[season] = {
            count: seasonCustomers.length,
            avgAmount: seasonCustomers.length > 0
                ? Math.round(seasonCustomers.reduce((sum, c) => sum + c.purchaseAmount, 0) / seasonCustomers.length)
                : 0
        };
    });

    return trends;
}

module.exports = {
    loadAndAnalyzeData,
    getCategoryStats,
    getTopItems,
    getSeasonalTrends
};
