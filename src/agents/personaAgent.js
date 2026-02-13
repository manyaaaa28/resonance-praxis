/**
 * Persona Agent for RESONANCE
 * 
 * Individual AI agent that embodies a customer persona
 * Supports Grok AI (xAI) and Gemini AI (Google)
 */

const { GoogleGenerativeAI } = require('@google/generative-ai');
const OpenAI = require('openai');

// Initialize AI clients based on environment
let geminiClient = null;
let grokClient = null;  // Now also supports Groq

function initializeClients() {
    const provider = process.env.AI_PROVIDER || 'gemini';

    if (provider === 'grok' && process.env.XAI_API_KEY) {
        // Support both xAI Grok and Groq (detect by key prefix)
        const isGroq = process.env.XAI_API_KEY.startsWith('gsk_');
        grokClient = new OpenAI({
            apiKey: process.env.XAI_API_KEY,
            baseURL: isGroq ? 'https://api.groq.com/openai/v1' : 'https://api.x.ai/v1'
        });
        console.log(`✅ ${isGroq ? 'Groq' : 'Grok'} AI client initialized`);
    }

    if (process.env.GEMINI_API_KEY) {
        geminiClient = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
        console.log('✅ Gemini AI client initialized');
    }

    if (!geminiClient && !grokClient) {
        console.warn('⚠️ No AI client initialized - will use fallback responses');
    }
}

// Initialize on module load
initializeClients();

/**
 * Create a persona agent
 * @param {string} type - Persona type (budget, premium, impulse, loyal)
 * @param {Object} persona - Persona profile with traits and prompt
 * @returns {Object} - Agent with evaluate method
 */
function createPersonaAgent(type, persona) {
    return {
        type,
        persona,

        /**
         * Evaluate a shopping scenario
         * @param {Object} scenario - { discount, category, brand, season }
         * @returns {Object} - { thought, decision, action, confidence }
         */
        async evaluate(scenario) {
            const { discount, category, brand, season } = scenario;

            // Build the evaluation prompt
            const userPrompt = buildEvaluationPrompt(type, persona, scenario);

            try {
                // Try AI providers in order of preference
                const provider = process.env.AI_PROVIDER || 'gemini';

                if (provider === 'grok' && grokClient) {
                    return await evaluateWithGrok(persona, userPrompt);
                } else if (geminiClient) {
                    return await evaluateWithGemini(persona, userPrompt);
                } else {
                    // Fallback to rule-based response
                    return generateRuleBasedResponse(type, scenario);
                }
            } catch (error) {
                console.error(`AI error for ${type}:`, error.message);
                return generateRuleBasedResponse(type, scenario);
            }
        }
    };
}

/**
 * Build evaluation prompt for the AI
 */
function buildEvaluationPrompt(type, persona, scenario) {
    const { discount, category, brand, season } = scenario;

    return `
You are shopping for ${category} products during ${season} season.
The store is offering ${discount}% discount on ${brand} brand items.

Based on your personality and shopping habits, share your honest thoughts about this offer in 1-2 sentences.
Then decide: Will you BUY, LEAVE the store, or HESITATE?

Respond in this exact JSON format:
{
    "thought": "Your internal monologue about this offer (1-2 sentences, first person, casual tone)",
    "decision": "buy" or "leave" or "hesitate",
    "action": "Brief action description (e.g., 'Added to cart', 'Walking away', 'Still thinking')",
    "confidence": 0.0 to 1.0
}

Be authentic to your persona. Sound human, not robotic.`;
}

/**
 * Evaluate using Grok/Groq AI
 */
async function evaluateWithGrok(persona, userPrompt) {
    // Use appropriate model based on provider
    const isGroq = process.env.XAI_API_KEY?.startsWith('gsk_');
    const model = isGroq ? 'llama-3.3-70b-versatile' : 'grok-beta';

    const completion = await grokClient.chat.completions.create({
        model,
        messages: [
            { role: 'system', content: persona.prompt },
            { role: 'user', content: userPrompt }
        ],
        temperature: 0.8,
        max_tokens: 200
    });

    const content = completion.choices[0]?.message?.content || '';
    return parseAIResponse(content);
}

/**
 * Evaluate using Gemini AI
 */
async function evaluateWithGemini(persona, userPrompt) {
    const model = geminiClient.getGenerativeModel({ model: 'gemini-1.5-flash' });

    const fullPrompt = `${persona.prompt}\n\n${userPrompt}`;

    const result = await model.generateContent(fullPrompt);
    const content = result.response.text();

    return parseAIResponse(content);
}

/**
 * Parse AI response to extract structured data
 */
function parseAIResponse(content) {
    try {
        // Try to extract JSON from response
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            const parsed = JSON.parse(jsonMatch[0]);
            return {
                thought: parsed.thought || 'Thinking about this offer...',
                decision: normalizeDecision(parsed.decision),
                action: parsed.action || 'Considering options',
                confidence: parseFloat(parsed.confidence) || 0.7
            };
        }
    } catch (e) {
        // If JSON parsing fails, extract from text
    }

    // Fallback: extract meaning from text
    const thought = content.split('\n')[0] || 'Evaluating this offer...';
    const decision = content.toLowerCase().includes('buy') ? 'buy'
        : content.toLowerCase().includes('leave') ? 'leave'
            : 'hesitate';

    return {
        thought: thought.replace(/["\{\}]/g, '').trim().slice(0, 150),
        decision,
        action: decision === 'buy' ? 'Added to cart' : decision === 'leave' ? 'Left store' : 'Still deciding',
        confidence: 0.6
    };
}

/**
 * Normalize decision string
 */
function normalizeDecision(decision) {
    if (!decision) return 'hesitate';
    const d = decision.toLowerCase().trim();
    if (d.includes('buy') || d.includes('purchase')) return 'buy';
    if (d.includes('leave') || d.includes('exit') || d.includes('no')) return 'leave';
    return 'hesitate';
}

/**
 * Generate rule-based response when AI is unavailable
 */
function generateRuleBasedResponse(type, scenario) {
    const { discount, category, brand } = scenario;

    const responses = {
        budget: {
            low: {
                thought: `Only ${discount}% off ${category}? I've seen better deals. I'll wait.`,
                decision: 'leave',
                action: '🚶 Left store',
                confidence: 0.8
            },
            medium: {
                thought: `${discount}% off ${category}! That's more like it. Adding to cart!`,
                decision: 'buy',
                action: '🛒 Added to cart',
                confidence: 0.75
            },
            high: {
                thought: `${discount}% OFF?! This is amazing! Buying multiple!`,
                decision: 'buy',
                action: '✅ Purchased',
                confidence: 0.95
            }
        },
        premium: {
            low: {
                thought: `Quality ${category} are worth full price. Purchasing now.`,
                decision: 'buy',
                action: '✅ Purchased',
                confidence: 0.85
            },
            medium: {
                thought: `Nice ${discount}% bonus, but I'd buy ${brand} quality anyway.`,
                decision: 'buy',
                action: '✅ Purchased',
                confidence: 0.8
            },
            high: {
                thought: `${discount}% off ${brand}? Such steep discounts worry me about quality...`,
                decision: 'hesitate',
                action: '🤔 Hesitating',
                confidence: 0.5
            }
        },
        impulse: {
            low: {
                thought: `Ooh, cute ${category}! *impulse adds to cart*`,
                decision: 'buy',
                action: '⚡ Impulse buy',
                confidence: 0.7
            },
            medium: {
                thought: `${category} on SALE? Buying two colors!`,
                decision: 'buy',
                action: '🛒 Multiple items',
                confidence: 0.85
            },
            high: {
                thought: `${discount}% OFF?! *adds everything to cart* NO REGRETS!`,
                decision: 'buy',
                action: '🎉 Shopping spree',
                confidence: 0.95
            }
        },
        loyal: {
            low: {
                thought: `Been buying ${brand} ${category} for years. Staying loyal!`,
                decision: 'buy',
                action: '💳 Used rewards',
                confidence: 0.9
            },
            medium: {
                thought: `Love the member discount on ${category}! Sharing with friends.`,
                decision: 'buy',
                action: '📧 Shared referral',
                confidence: 0.85
            },
            high: {
                thought: `Amazing ${category} deal for loyal customers! Buying gifts for everyone!`,
                decision: 'buy',
                action: '🎁 Bought gifts',
                confidence: 0.95
            }
        }
    };

    // Determine discount level
    const level = discount < 20 ? 'low' : discount < 40 ? 'medium' : 'high';

    return responses[type]?.[level] || {
        thought: `Considering this ${category} offer...`,
        decision: 'hesitate',
        action: 'Thinking',
        confidence: 0.5
    };
}

module.exports = {
    createPersonaAgent,
    initializeClients
};
