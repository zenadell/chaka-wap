import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import https from 'https';
import http from 'http';
import { Server } from 'socket.io';
import makeWASocket, {
    useMultiFileAuthState,
    makeCacheableSignalKeyStore,
    DisconnectReason,
    delay,
    Browsers,
    downloadMediaMessage,
    fetchLatestBaileysVersion
} from '@whiskeysockets/baileys';
import qrcode from 'qrcode';
import { createRequire } from 'module';
import { GoogleGenerativeAI } from "@google/generative-ai";
import fs from 'fs';
import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import path from 'path';
import { fileURLToPath } from 'url';
import { OpenRouter } from "@openrouter/sdk";
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import passport from 'passport';
import { Strategy as GoogleStrategy } from 'passport-google-oauth20';
import { v4 as uuidv4 } from 'uuid';

// --- SECURITY PROTOCOLS ---
const JWT_SECRET = process.env.JWT_SECRET || "chaka_super_secret_dev_key_2026";
const ADMIN_EMAIL = "timtemple2024@gmail.com";

// --- GLOBAL STABILITY HANDLERS ---
process.on('unhandledRejection', (reason, promise) => {
    console.error('💥 UNHANDLED REJECTION:', reason);
});
process.on('uncaughtException', (err) => {
    console.error('💥 UNCAUGHT EXCEPTION:', err);
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
import { pipeline } from '@xenova/transformers'; // NEW: Local AI

// --- CONFIGURATION ---
const MY_NAME = "Temple";
const require = createRequire(import.meta.url);

// --- GLOBAL STATE ---
let API_KEYS;
let ACTIVE_API;
let QWEN_ENDPOINT;
let OPENROUTER_API_KEY;
let CHAKA_MODEL = 'chaka-medium';
const CHAKA_MODELS = ['chaka-ultimate', 'chaka-high', 'chaka-medium', 'chaka-low'];
let db;

// Ensure data directory exists for persistent storage
if (!fs.existsSync('./data')) {
    fs.mkdirSync('./data');
}

function initGlobalState() {
    API_KEYS = (process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || "AIzaSyATiIO8ouylB0mwhueHgu05gO2HpJaj3V4").split(',');
    ACTIVE_API = process.env.ACTIVE_API || 'openrouter';
    QWEN_ENDPOINT = process.env.QWEN_ENDPOINT || "http://localhost:8000/api/chat";
    OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || "";
    
    console.log(`\n=================================================`);
    console.log(`⚙️ Engine: ${ACTIVE_API.toUpperCase()}`);
    console.log(`🔌 Qwen Endpoint: ${QWEN_ENDPOINT}`);
    console.log(`🔑 Gemini Keys: ${API_KEYS.length}`);
    console.log(`🔑 OpenRouter Key: ${OPENROUTER_API_KEY ? 'Configured' : 'Missing'}`);
    console.log(`=================================================\n`);
}

initGlobalState();

let localEmbedder = null; // Holds the local AI model
const AI_REPLY_TRACKER = new Map(); // Tracks last responded message ID per contact
const BURST_TRACKER = new Map(); // contactId -> [{ts, tokens}]
const SPEND_TRACKER = new Map(); // sessionId -> [{minute, tokens}]
const RECONNECT_ATTEMPTS = new Map(); // sessionId -> count

const SLANG_MAP = {
    "hw far": "how far",
    "wsg": "what is good",
    "typeshit": "I agree",
    "real shit": "true story",
    "fvck": "fuck",
    "nfs": "no funny shit",
    "fr": "for real",
    "ngl": "not gonna lie",
    "istg": "I swear to god"
};

function expandSlang(text) {
    if (!text) return "";
    let expanded = text.toLowerCase();
    for (const [slang, standard] of Object.entries(SLANG_MAP)) {
        const regex = new RegExp(`\\b${slang}\\b`, 'g');
        expanded = expanded.replace(regex, standard);
    }
    return expanded;
}

function trackTokenSpend(sessionId, tokens) {
    const now = new Date();
    const minuteKey = now.getHours() * 60 + now.getMinutes();
    let spends = SPEND_TRACKER.get(sessionId) || [];
    spends = spends.filter(s => s.minute >= minuteKey - 1); // Keep last 2 mins
    const current = spends.find(s => s.minute === minuteKey);
    if (current) current.tokens += tokens;
    else spends.push({ minute: minuteKey, tokens });
    SPEND_TRACKER.set(sessionId, spends);
}

function getCurrentMinuteSpend(sessionId) {
    const now = new Date();
    const minuteKey = now.getHours() * 60 + now.getMinutes();
    const spends = SPEND_TRACKER.get(sessionId) || [];
    const current = spends.find(s => s.minute === minuteKey);
    return current?.tokens || 0;
}

async function isBurstMode(contactId) {
    const now = Date.now();
    const window = BURST_TRACKER.get(contactId) || [];
    const recent = window.filter(t => now - t.ts < 120000); // 2 mins
    return recent.length >= 3;
}

function recordMessage(contactId, tokens) {
    const window = BURST_TRACKER.get(contactId) || [];
    window.push({ ts: Date.now(), tokens });
    BURST_TRACKER.set(contactId, window.slice(-10));
}


// --- DATABASE INITIALIZATION (SQLITE) ---
async function initDB() {
    db = await open({
        filename: './data/chaka_data.db', // THIS IS YOUR ENTIRE UNLIMITED DATABASE FILE!
        driver: sqlite3.Database
    });

    await db.exec(`
        -- Multi-Tenant Core Tables
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            display_name TEXT,
            role TEXT DEFAULT 'user',
            max_sessions INTEGER DEFAULT 4,
            created_at INTEGER,
            last_login INTEGER,
            google_id TEXT
        );
        CREATE TABLE IF NOT EXISTS request_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            engine TEXT,
            model TEXT,
            status TEXT,
            error_message TEXT,
            timestamp INTEGER
        );
        
        -- Existing Tables (will be altered below for multi-tenant support)
        CREATE TABLE IF NOT EXISTS global_settings (
            id TEXT,
            user_id TEXT DEFAULT 'admin',
            api_keys TEXT,
            custom_prompt TEXT,
            master_auto_reply INTEGER DEFAULT 1,
            active_api TEXT DEFAULT 'gemini',
            chaka_model TEXT DEFAULT 'chaka-medium',
            PRIMARY KEY (id, user_id)
        );
        CREATE TABLE IF NOT EXISTS contacts (
            session_id TEXT,
            contact_id TEXT,
            user_id TEXT DEFAULT 'admin',
            name TEXT,
            jid TEXT,
            last_active INTEGER,
            auto_reply INTEGER DEFAULT 1,
            custom_prompt TEXT,
            PRIMARY KEY (session_id, contact_id)
        );
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT,
            contact_id TEXT,
            user_id TEXT DEFAULT 'admin',
            text TEXT,
            sender TEXT,
            timestamp INTEGER,
            date TEXT,
            is_from_me INTEGER,
            embedding TEXT
        );
        CREATE TABLE IF NOT EXISTS stickers (
            file_sha256 TEXT PRIMARY KEY,
            file_path TEXT,
            session_id TEXT,
            contact_id TEXT,
            user_id TEXT DEFAULT 'admin',
            usage_count INTEGER DEFAULT 1,
            last_used INTEGER
        );
    `);

    // --- MIGRATION: ADD USER_ID COLUMNS (IGNORES IF EXIST) ---
    try { await db.exec(`ALTER TABLE global_settings ADD COLUMN user_id TEXT DEFAULT 'admin'`); } catch (e) { }
    // Note: SQLite ALTER TABLE ADD COLUMN cannot define composite Primary Keys retroactively, 
    // so we rely on session scoping. It's safe since session IDs are globally unique per user via auth\_baileys_{uid}_{sid}.
    try { await db.exec(`ALTER TABLE global_settings ADD COLUMN active_api TEXT DEFAULT 'gemini'`); } catch (e) { }
    try { await db.exec(`ALTER TABLE global_settings ADD COLUMN chaka_model TEXT DEFAULT 'chaka-medium'`); } catch (e) { }
    try { await db.exec(`ALTER TABLE contacts ADD COLUMN user_id TEXT DEFAULT 'admin'`); } catch (e) { }
    try { await db.exec(`ALTER TABLE messages ADD COLUMN user_id TEXT DEFAULT 'admin'`); } catch (e) { }
    try { await db.exec(`ALTER TABLE stickers ADD COLUMN user_id TEXT DEFAULT 'admin'`); } catch (e) { }

    // --- ADMIN SEEDING ---
    const adminExists = await db.get(`SELECT id FROM users WHERE email = ?`, [ADMIN_EMAIL]);
    let adminId = 'admin';
    if (!adminExists) {
        const hash = await bcrypt.hash("admin123", 10);
        await db.run(`INSERT INTO users (id, email, password_hash, display_name, role, created_at) VALUES (?, ?, ?, ?, ?, ?)`, 
            ['admin', ADMIN_EMAIL, hash, 'Tim Temple', 'admin', Math.floor(Date.now() / 1000)]);
    } else {
        adminId = adminExists.id;
    }

    // --- ORPHAN DATA ADOPTION (Migrate old single-user data to Admin) ---
    await db.run(`UPDATE global_settings SET user_id = ? WHERE user_id IS NULL OR user_id = 'admin'`, [adminId]);
    await db.run(`UPDATE contacts SET user_id = ? WHERE user_id IS NULL OR user_id = 'admin'`, [adminId]);
    await db.run(`UPDATE messages SET user_id = ? WHERE user_id IS NULL OR user_id = 'admin'`, [adminId]);
    await db.run(`UPDATE stickers SET user_id = ? WHERE user_id IS NULL OR user_id = 'admin'`, [adminId]);

    // Ensure global settings exist for Admin
    await db.run(`INSERT OR IGNORE INTO global_settings (id, user_id, master_auto_reply, active_api, chaka_model) VALUES ('settings', ?, 1, 'gemini', 'chaka-medium')`, [adminId]);
    await loadGlobalConfig(adminId); // Load the admin's config just for legacy compatibility in memory, though we should transition to per-user active_apis.
    console.log("🗄️ Multi-Tenant SQLite Database Initialized Successfully.");
}

async function loadGlobalConfig() {
    try {
        const row = await db.get(`SELECT * FROM global_settings WHERE id = 'settings'`);
        if (row) {
            if (row.api_keys) {
                const dbKeys = row.api_keys.split(',').map(k => k.trim()).filter(k => k.length > 5);
                if (dbKeys.length > 0) API_KEYS = dbKeys;
            }
            if (row.active_api) ACTIVE_API = row.active_api;
            if (row.chaka_model) CHAKA_MODEL = row.chaka_model;
        }
    } catch (e) {
        console.warn("Failed to load global config from DB (using defaults):", e.message);
    }
}

// --- LOCAL AI MEMORY ENGINE (100% FREE & UNLIMITED) ---
async function initLocalAI() {
    if (process.env.FLY_APP_NAME) {
        console.log("☁️ Running on Fly.io: Skipping memory-intensive Local AI Boot to save RAM.");
        return;
    }
    console.log("🧠 Booting Local AI Memory Engine (Downloads tiny model on first run)...");
    try {
        // This loads the industry-standard MiniLM model directly into your server's RAM
        localEmbedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        console.log("✅ Local AI Ready. Memory vectors are now generated locally at zero cost.");
    } catch (e) {
        console.error("🔥 Local AI Boot Failed:", e.message);
    }
}

async function generateLocalEmbedding(text) {
    if (!localEmbedder || !text || text.trim().length === 0) return null;
    try {
        const output = await localEmbedder(text, { pooling: 'mean', normalize: true });
        return Buffer.from(new Float32Array(output.data).buffer);
    } catch (error) {
        console.error("🔥 Local Embedding Failed:", error);
        return null;
    }
}

// --- AI SETUP ---
async function getRotatedModel(userId, modelName = "gemini-3.1-pro-preview") {
    const row = await db.get(`SELECT api_keys FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);
    const keys = (row && row.api_keys) ? row.api_keys.split(',').map(k => k.trim()).filter(k => k.length > 5) : ["AIzaSyATiIO8ouylB0mwhueHgu05gO2HpJaj3V4"];
    const randomKey = keys[Math.floor(Math.random() * keys.length)];
    const genAI = new GoogleGenerativeAI(randomKey);

    if (modelName === "text-embedding-004" || modelName === "embedding-001") {
        return genAI.getGenerativeModel({ model: "embedding-001" });
    } else {
        return genAI.getGenerativeModel({ model: modelName, generationConfig: { temperature: 0.9 } });
    }
}

async function fetchUserStyleSamples(userId, sessionId, contactId) {
    try {
        console.log(`[${sessionId}] 🎭 Analyzing user writing style for ${contactId}...`);

        // 1. Try to get samples specifically from this contact
        let samples = await db.all(
            `SELECT text FROM messages 
             WHERE session_id = ? AND contact_id = ? AND user_id = ? AND is_from_me = 1 
             ORDER BY timestamp DESC LIMIT 8`,
            [sessionId, contactId, userId]
        );

        // 2. Fallback: If no samples for this contact, get general recent style from this session
        if (!samples || samples.length < 3) {
            console.log(`[${sessionId}] 🎭 Limited contact samples, pulling general session style...`);
            samples = await db.all(
                `SELECT text FROM messages 
                 WHERE session_id = ? AND user_id = ? AND is_from_me = 1 
                 ORDER BY timestamp DESC LIMIT 10`,
                [sessionId, userId]
            );
        }

        if (!samples || samples.length === 0) return "";

        // 3. Clean up samples to avoid repeating bot artifacts or boring loops
        const cleanedSamples = (samples || [])
            .map(s => s.text)
            .filter(t => t && !t.includes('hey boy') && !t.includes('i gots ya man') && !t.includes('typeshit')) // Filter out the loops
            .slice(0, 5);

        if (cleanedSamples.length === 0) return null;
        return cleanedSamples.join('\n');
    } catch (e) {
        console.error("Style sample error:", e);
        return null;
    }
}

async function describeImage(userId, sessionId, imageBuffer, chatContext = "") {
    try {
        console.log(`[${sessionId}] 👁️ Analyzing incoming image with Gemini Vision...`);
        // Vision requires 1.5-flash for binary inline stability, we use 2.5-flash for the main brain
        const model = await getRotatedModel(userId, "gemini-3.1-pro-preview");

        let prompt = `You are an expert at understanding the "occasion" or "reason" why someone sends an image in a casual WhatsApp chat.
        
        RECENT CHAT CONTEXT:
        ${chatContext}
        
        TASK:
        Describe this image and, most importantly, identify WHY the user sent it based on the context above. 
        Be extremely concise (max 20 words). Stay casual.`;

        const result = await model.generateContent([
            prompt,
            {
                inlineData: {
                    data: imageBuffer.toString("base64"),
                    mimeType: "image/jpeg",
                },
            },
        ]);

        const description = result.response.text().trim();
        console.log(`[${sessionId}] 👁️ Contextual Vision Insight: ${description}`);
        return description;
    } catch (e) {
        console.error("Vision Analysis Failed:", e.message);
        return "an image";
    }
}

function parseEmbedding(data) {
    if (!data) return new Float32Array(0);
    try {
        if (Buffer.isBuffer(data)) {
            return new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
        }
        if (typeof data === 'string' && data.startsWith('[')) {
            return new Float32Array(JSON.parse(data));
        }
        return new Float32Array(0);
    } catch (e) {
        return new Float32Array(0);
    }
}

function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length === 0 || vecA.length !== vecB.length) return -1;
    let dotProduct = 0, normA = 0, normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    return isNaN(similarity) ? -1 : similarity;
}


// --- CHAKA API PLATFORM INTEGRATION ---
async function generateChakaResponse(userId, prompt, modelOverride = null) {
    try {
        const url = "https://chaka-model.onrender.com/api/chat";
        const headers = { 'Content-Type': 'application/json', 'X-Chaka-API-Key': 'Chaka_Supreme_Access' };
        
        let model = modelOverride;
        if (!model) {
             const row = await db.get(`SELECT chaka_model FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);
             model = (row && row.chaka_model) ? row.chaka_model : 'chaka-low';
        }

        console.log(`🔌 Calling Chaka API -> Model: ${model}`);
        const response = await fetch(url, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({ message: prompt, model: model })
        });

        if (!response.ok) throw new Error(`Chaka API responded with status: ${response.status}`);

        const data = await response.json();
        const finalReply = data.response || data.reply || data.message || data.text || data.choices?.[0]?.message?.content || JSON.stringify(data);
        return finalReply;
    } catch (error) {
        console.error("🔥 Chaka API Fetch Error:", error.message);
        throw error;
    }
}

// --- OPENROUTER LLM INTEGRATION ---
async function tryOpenRouterFailover(userId, sessionId, systemPrompt, userPrompt) {
    if (!OPENROUTER_API_KEY) {
        console.warn(`[${sessionId}] ⚠️ OpenRouter API Key missing!`);
        return null;
    }

    console.log(`[${sessionId}] 🔄 Requesting OpenRouter (Qwen 3 Coder)...`);
    
    // Fallback prompt combination
    const combinedPrompt = `${systemPrompt}\n\n${userPrompt}`;

    try {
        const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
                'Content-Type': 'application/json',
                'HTTP-Referer': 'http://localhost:3000',
                'X-OpenRouter-Title': 'WhatsApp Crawler AI'
            },
            body: JSON.stringify({
                model: 'qwen/qwen3-coder:free',
                messages: [{ role: 'user', content: combinedPrompt }]
            })
        });

        const data = await response.json();
        
        if (data && data.choices && data.choices.length > 0) {
            return data.choices[0].message.content.trim();
        } else if (data.error) {
            console.warn(`[${sessionId}] ⚠️ OpenRouter Error: ${data.error.message} (Code: ${data.error.code})`);
        }
    } catch (error) {
        console.error(`[${sessionId}] 🔥 OpenRouter Fetch Error:`, error.message);
    }
    
    return null;
}

// --- LOCAL QWEN LLM INTEGRATION ---
async function generateQwenResponse(systemPrompt, userPrompt) {
    return new Promise((resolve, reject) => {
        const urlString = QWEN_ENDPOINT.trim();
        const data = JSON.stringify({ 
            system_prompt: systemPrompt,
            user_prompt: userPrompt 
        });
        
        console.log(`🔌 Requesting Colab Engine (Llama 3.1/3.2) -> ${urlString}`);
        
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(data),
                'ngrok-skip-browser-warning': 'true',
                'User-Agent': 'ChakaWorker/1.0'
            },
            timeout: 90000,
            agent: false // CRITICAL: Disables socket pooling/reuse which fixes 'socket hang up' on ngrok
        };

        const client = urlString.startsWith('https') ? https : http;
        const req = client.request(urlString, options, (res) => {
            let body = '';
            res.on('data', (chunk) => body += chunk);
            res.on('end', () => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                    try {
                        const parsed = JSON.parse(body);
                        resolve(parsed.response || parsed.reply || "");
                    } catch (e) {
                        reject(new Error(`Failed to parse AI response: ${body.substring(0, 100)}`));
                    }
                } else {
                    reject(new Error(`Colab Engine responded with status ${res.statusCode}: ${body.substring(0, 100)}`));
                }
            });
        });

        req.on('error', (e) => {
            console.error("🔥 Colab Connection Error:", e.message);
            reject(e);
        });
        
        req.on('timeout', () => {
            req.destroy();
            reject(new Error("Colab Engine timed out (90s)"));
        });

        req.write(data);
        req.end();
    });
}

async function orchestrateAIResponse(userId, sessionId, systemPrompt, userPrompt) {
    let attempts = 0;
    const fullLegacyPrompt = `${systemPrompt}\n\n${userPrompt}`; // Fallback for Gemini/Chaka

    const row = await db.get(`SELECT active_api FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);
    const activeApi = (row && row.active_api) ? row.active_api : 'gemini';

    while (true) {
        attempts++;
        console.log(`[${sessionId}] 🛰️ AI Orchestrator: Try #${attempts}`);

        try {
            if (activeApi === 'qwen') {
                // LLAMA 3.1 (Labelled Qwen/Local)
                const result = await tryQwenFailover(userId, sessionId, systemPrompt, userPrompt);
                if (result) return result;

                // FALLBACK ONLY TO CHAKA IF LLAMA IS DOWN
                console.warn(`[${sessionId}] Llama 3.1 Offline, falling back to Chaka...`);
                const chakaResult = await tryChakaFailover(userId, sessionId, fullLegacyPrompt);
                if (chakaResult) return chakaResult;

            } else if (activeApi === 'openrouter') {
                const result = await tryOpenRouterFailover(userId, sessionId, systemPrompt, userPrompt);
                if (result) return result;
                
                console.warn(`[${sessionId}] OpenRouter Offline, falling back to Chaka...`);
                const chakaResult = await tryChakaFailover(userId, sessionId, fullLegacyPrompt);
                if (chakaResult) return chakaResult;

            } else if (activeApi === 'gemini') {
                const result = await tryGeminiFailover(userId, sessionId, fullLegacyPrompt);
                if (result) return result;
                const chakaResult = await tryChakaFailover(userId, sessionId, fullLegacyPrompt);
                if (chakaResult) return chakaResult;
            } else if (activeApi === 'chaka') {
                const result = await tryChakaFailover(userId, sessionId, fullLegacyPrompt);
                if (result) return result;

                console.warn(`[${sessionId}] Chaka Offline, falling back to Gemini...`);
                const geminiResult = await tryGeminiFailover(userId, sessionId, fullLegacyPrompt);
                if (geminiResult) return geminiResult;
            } else {
                const result = await tryChakaFailover(userId, sessionId, fullLegacyPrompt);
                if (result) return result;
                const geminiResult = await tryGeminiFailover(userId, sessionId, fullLegacyPrompt);
                if (geminiResult) return geminiResult;
            }

            // If we are here, EVERYTHING failed.
            console.warn(`[${sessionId}] 🚨 CRITICAL: All providers/models failed. Entering 2-minute cooldown...`);
            io.emit('log', { sessionId, msg: "🚨 All AI systems exhausted. Cooling down for 2 mins..." });
            await delay(120000); // 2 minute cooldown
        } catch (e) {
            console.error("Orchestrator Loop Error:", e);
            await delay(5000);
        }
    }
}

async function tryQwenFailover(userId, sessionId, systemPrompt, userPrompt) {
    console.log(`[${sessionId}] 🔄 Trying Local Llama 3.1 (Max 3 Retries)...`);
    let retries = 0;
    while (retries < 3) {
        try {
            const reply = await generateQwenResponse(systemPrompt, userPrompt);
            if (reply && reply.length > 0) return reply;
        } catch (e) {
            retries++;
            console.warn(`[${sessionId}] ⚠️ Qwen Attempt #${retries} Failed: ${e.message}`);
            if (retries < 3) {
                console.log(`[${sessionId}] ⏳ Retrying Qwen in ${retries * 5}s...`);
                await delay(retries * 5000);
            }
        }
    }
    return null;
}

async function tryChakaFailover(userId, sessionId, prompt) {
    console.log(`[${sessionId}] 🔄 Starting Chaka Model Rotation...`);
    for (const model of CHAKA_MODELS) {
        try {
            const reply = await generateChakaResponse(userId, prompt, model);
            if (reply && reply.length > 0) return reply;
        } catch (e) {
            console.warn(`[${sessionId}] ⚠️ Chaka ${model} Failed: ${e.message}`);
            await delay(3000); // Mandatory 3s delay
        }
    }
    return null;
}

async function tryGeminiFailover(userId, sessionId, prompt) {
    console.log(`[${sessionId}] 🔄 Starting Gemini Key Rotation & 3.1 Model Hunt...`);
    const modelsToTry = ["gemini-3.1-pro-preview"];
    
    const row = await db.get(`SELECT api_keys FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);
    const keys = (row && row.api_keys) ? row.api_keys.split(',').map(k => k.trim()).filter(k => k.length > 5) : ["AIzaSyATiIO8ouylB0mwhueHgu05gO2HpJaj3V4"];
    if (keys.length === 0) return null;

    for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        const genAI = new GoogleGenerativeAI(key);
        
        for (const modelName of modelsToTry) {
            try {
                console.log(`[${sessionId}] 🛠️ Testing Gemini -> Key #${i + 1} | Model: ${modelName}`);
                const model = genAI.getGenerativeModel({ model: modelName });
                const result = await model.generateContent(prompt);
                const text = result.response.text();
                if (text && text.length > 0) return text.trim();
            } catch (e) {
                console.warn(`[${sessionId}] ⚠️ Gemini ${modelName} (Key #${i + 1}) Failed: ${e.message.split('\n')[0]}`);
                await delay(2000); // Small gap between model tries
            }
        }
    }
    return null;
}

// --- SERVER SETUP & AUTHENTICATION ---
const app = express();
app.set('trust proxy', 1); // Trust Fly.io proxy for HTTPS redirects
app.use(express.json());
const server = createServer(app);
const io = new Server(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});
app.use(express.static('public'));

// Server-side redirect for root to login
app.get('/', (req, res) => {
    res.redirect('/login.html');
});

// Health check for Fly.io
app.get('/health', (req, res) => {
    res.status(200).send('OK');
});

// --- GOOGLE OAUTH STRATEGY ---
passport.use(new GoogleStrategy({
    clientID: process.env.GOOGLE_CLIENT_ID || 'dummy_client_id',
    clientSecret: process.env.GOOGLE_CLIENT_SECRET || 'dummy_client_secret',
    callbackURL: "/auth/google/callback"
  },
  async function(accessToken, refreshToken, profile, cb) {
    try {
        const email = profile.emails[0].value;
        const displayName = profile.displayName;
        
        let user = await db.get('SELECT * FROM users WHERE email = ?', [email]);
        if (!user) {
            const userId = uuidv4();
            const role = email === 'timtemple2024@gmail.com' ? 'admin' : 'user';
            const hash = await bcrypt.hash(uuidv4(), 10);
            await db.run(
                `INSERT INTO users (id, email, password_hash, display_name, role, created_at) VALUES (?, ?, ?, ?, ?, ?)`,
                [userId, email, hash, displayName, role, Math.floor(Date.now() / 1000)]
            );
            await db.run(
                `INSERT OR IGNORE INTO global_settings (id, user_id, master_auto_reply, active_api, chaka_model) VALUES ('settings', ?, 1, 'openrouter', 'chaka-medium')`, 
                [userId]
            );
            user = { id: userId, email, display_name: displayName, role };
        } else if (email === 'timtemple2024@gmail.com' && user.role !== 'admin') {
            await db.run('UPDATE users SET role = "admin" WHERE id = ?', [user.id]);
            user.role = 'admin';
        }
        return cb(null, user);
    } catch (e) {
        return cb(e, null);
    }
  }
));

app.get('/auth/google', passport.authenticate('google', { scope: ['profile', 'email'], session: false }));

app.get('/auth/google/callback', passport.authenticate('google', { session: false, failureRedirect: '/login.html?error=oauth_failed' }),
  function(req, res) {
    const user = req.user;
    const token = jwt.sign({ id: user.id, email: user.email, role: user.role }, JWT_SECRET, { expiresIn: '7d' });
    
    res.send(`
        <script>
            localStorage.setItem('token', '${token}');
            localStorage.setItem('user', JSON.stringify(${JSON.stringify({ id: user.id, email: user.email, displayName: user.display_name, role: user.role })}));
            window.location.href = '/';
        </script>
    `);
  });

// JWT Middleware
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    if (!token) return res.status(401).json({ error: "Access Denied" });
    jwt.verify(token, JWT_SECRET, (err, user) => {
        if (err) return res.status(403).json({ error: "Invalid Token" });
        req.user = user;
        next();
    });
};

// --- AUTH API ENDPOINTS ---
app.post('/api/register', async (req, res) => {
    try {
        const { email, password, displayName } = req.body;
        if (!email || !password) return res.status(400).json({ error: "Email and password required" });
        
        const existing = await db.get('SELECT id FROM users WHERE email = ?', [email]);
        if (existing) return res.status(400).json({ error: "Email already registered" });

        const hash = await bcrypt.hash(password, 10);
        const userId = uuidv4();
        const role = email === 'timtemple2024@gmail.com' ? 'admin' : 'user';
        
        await db.run(
            `INSERT INTO users (id, email, password_hash, display_name, role, created_at) VALUES (?, ?, ?, ?, ?, ?)`,
            [userId, email, hash, displayName || email.split('@')[0], role, Math.floor(Date.now() / 1000)]
        );

        // Auto-seed global settings for the new user so the dashboard doesn't crash
        await db.run(
            `INSERT OR IGNORE INTO global_settings (id, user_id, master_auto_reply, active_api, chaka_model) VALUES ('settings', ?, 1, 'openrouter', 'chaka-medium')`, 
            [userId]
        );

        const token = jwt.sign({ id: userId, email, role: role }, JWT_SECRET, { expiresIn: '7d' });
        res.json({ token, user: { id: userId, email, displayName: displayName || email.split('@')[0], role: role } });
    } catch (e) {
        console.error("Register error:", e);
        res.status(500).json({ error: "Internal server error" });
    }
});

app.post('/api/login', async (req, res) => {
    try {
        const { email, password } = req.body;
        const user = await db.get('SELECT * FROM users WHERE email = ?', [email]);
        if (!user || !(await bcrypt.compare(password, user.password_hash))) {
            return res.status(401).json({ error: "Invalid email or password" });
        }
        
        await db.run('UPDATE users SET last_login = ? WHERE id = ?', [Math.floor(Date.now() / 1000), user.id]);
        
        const token = jwt.sign({ id: user.id, email: user.email, role: user.role }, JWT_SECRET, { expiresIn: '7d' });
        res.json({ token, user: { id: user.id, email: user.email, displayName: user.display_name, role: user.role } });
    } catch (e) {
        console.error("Login error:", e);
        res.status(500).json({ error: "Internal server error" });
    }
});

app.get('/api/me', authenticateToken, async (req, res) => {
    const user = await db.get('SELECT id, email, display_name, role, max_sessions FROM users WHERE id = ?', [req.user.id]);
    res.json(user);
});

// --- ADMIN API ENDPOINTS ---
app.get('/api/admin/accounts', authenticateToken, async (req, res) => {
    if (req.user.role !== 'admin' && req.user.email !== 'timtemple2024@gmail.com') return res.status(403).json({ error: "Forbidden" });
    const users = await db.all(`
        SELECT u.id, u.email, u.display_name, u.role, u.created_at, u.last_login,
        (SELECT COUNT(DISTINCT session_id) FROM contacts WHERE user_id = u.id) as session_count
        FROM users u
    `);
    res.json(users);
});

app.get('/api/admin/stats', authenticateToken, async (req, res) => {
    if (req.user.role !== 'admin' && req.user.email !== 'timtemple2024@gmail.com') return res.status(403).json({ error: "Forbidden" });
    const totalUsers = await db.get('SELECT COUNT(*) as count FROM users');
    const totalSessions = await db.get('SELECT COUNT(DISTINCT session_id) FROM contacts');
    const totalMessages = await db.get('SELECT COUNT(*) as count FROM messages');
    const failedRequests = await db.get('SELECT COUNT(*) as count FROM request_logs WHERE status = "failed"');
    
    res.json({
        totalUsers: totalUsers.count,
        totalSessions: totalSessions['COUNT(DISTINCT session_id)'],
        totalMessages: totalMessages.count,
        failedRequests: failedRequests ? failedRequests.count : 0
    });
});

// Socket.IO JWT Authentication Middleware
io.use((socket, next) => {
    const token = socket.handshake.auth.token;
    if (!token) return next(new Error("Authentication error: No token provided"));
    
    jwt.verify(token, JWT_SECRET, (err, decoded) => {
        if (err) return next(new Error("Authentication error: Invalid or expired token"));
        socket.user = decoded; // Bind user data to socket
        next();
    });
});

const sessions = new Map();

// --- PERSISTENCE: RECOVER SESSIONS FROM DISK ON BOOT ---
function loadPersistedSessions() {
    try {
        const dataDir = './data';
        if (!fs.existsSync(dataDir)) return;
        const files = fs.readdirSync(dataDir);
        const folders = files.filter(f => f.startsWith('auth_baileys_'));
        console.log(`[SYSTEM] Found ${folders.length} archived sessions on disk. Restoring memory placeholders...`);
        folders.forEach(f => {
            const globalId = f.replace('auth_baileys_', '');
            if (!sessions.has(globalId)) {
                sessions.set(globalId, { isConnected: false, connectionState: 'disconnected', handshakeTimer: null });
            }
        });
    } catch (e) { console.error("Session recovery failed:", e); }
}
loadPersistedSessions();

function getSafeTimestamp(ts) {
    if (!ts) return Math.floor(Date.now() / 1000);
    if (typeof ts === 'object' && ts.low) return ts.low;
    if (typeof ts === 'object' && ts.toNumber) return ts.toNumber();
    return Number(ts);
}

function extractContent(msg) {
    if (!msg) return { type: 'unknown', text: null };

    // 1. UNWRAP DEEP MESSAGES FIRST (Ephemeral, View Once, Document Wrappers)
    const deepMsg = msg.viewOnceMessage?.message || msg.ephemeralMessage?.message || msg.documentWithCaptionMessage?.message || msg.viewOnceMessageV2?.message || msg.viewOnceMessageV2Extension?.message;
    if (deepMsg) return extractContent(deepMsg);

    // 2. Handle Edited Messages (ProtocolMessage)
    if (msg.protocolMessage && msg.protocolMessage.type === 14) { // 14 = MESSAGE_EDIT
        const edited = msg.protocolMessage.editedMessage;
        const editedText = edited?.conversation || edited?.extendedTextMessage?.text;
        if (editedText) return { type: 'edit', text: editedText, targetId: msg.protocolMessage.key?.id };
    }

    // 3. Handle Images
    if (msg.imageMessage) {
        return {
            type: 'image',
            text: msg.imageMessage.caption || null,
            msg: msg.imageMessage
        };
    }

    // 4. Handle Stickers
    if (msg.stickerMessage) {
        return {
            type: 'sticker',
            text: '[STICKER]',
            sha256: msg.stickerMessage.fileSha256 ? Buffer.from(msg.stickerMessage.fileSha256).toString('hex') : null,
            msg: msg.stickerMessage
        };
    }

    // 5. Handle Regular Text
    const text = msg.conversation || msg.extendedTextMessage?.text || msg.videoMessage?.caption || msg.documentMessage?.caption;
    if (text) return { type: 'text', text: text };

    return { type: 'junk', text: null };
}

function createStore() {
    const data = { messages: {} };
    return {
        data,
        bind: (ev) => {
            ev.on('messages.upsert', ({ messages }) => {
                for (const msg of messages) {
                    const jid = msg.key.remoteJid;
                    if (!data.messages[jid]) data.messages[jid] = [];
                    if (!data.messages[jid].some(m => m.key.id === msg.key.id)) data.messages[jid].push(msg);
                    if (data.messages[jid].length > 1000) data.messages[jid] = data.messages[jid].slice(-1000);
                }
            });
        },
        loadMessage: async (jid, id) => data.messages[jid]?.find(m => m.key.id === id)
    };
}

async function saveMessageToDB(userId, sessionId, msg, forceSkipEmbedding = false) {
    try {
        if (!msg) return null;
        const content = extractContent(msg.message);
        const jid = msg.key.remoteJid;

        if (!content.text && content.type !== 'sticker' && content.type !== 'image') {
            if (!forceSkipEmbedding) console.log(`[${sessionId}] ⏭ Skipping: No text content found.`);
            return null;
        }

        if (jid === 'status@broadcast' || jid.includes('@g.us')) {
            if (!forceSkipEmbedding) console.log(`[${sessionId}] ⏭ Skipping: Non-personal chat (${jid})`);
            return null;
        }

        const session = sessions.get(sessionId);
        const cleanId = jid.replace(/[^a-zA-Z0-9]/g, "_");

        // --- NEW: VISION ENGINE ---
        if (content.type === 'image') {
            try {
                console.log(`[${sessionId}] 👁️ Image detected. Fetching context for occasion analysis...`);

                // Fetch last 5 messages for context
                const recentContextRows = await db.all(
                    `SELECT sender, text FROM messages WHERE session_id = ? AND contact_id = ? ORDER BY timestamp DESC LIMIT 5`,
                    [sessionId, cleanId]
                );
                const chatContext = recentContextRows.reverse().map(m => `${m.sender}: ${m.text}`).join("\n");

                const buffer = await downloadMediaMessage(msg, 'buffer', {});
                if (buffer) {
                    const description = await describeImage(userId, sessionId, buffer, chatContext);
                    content.text = content.text ? `[IMAGE: ${description}] Caption: ${content.text}` : `[IMAGE: ${description}]`;
                }
            } catch (e) {
                console.error("Vision trigger failed:", e.message);
                content.text = content.text ? `[IMAGE] ${content.text}` : `[IMAGE]`;
            }
        }

        // Handle Sticker Downloading
        if (content.type === 'sticker' && content.sha256) {
            const stickerPath = path.join(__dirname, 'public', 'stickers', `${content.sha256}.webp`);
            if (!fs.existsSync(stickerPath)) {
                try {
                    console.log(`[${sessionId}] 📥 Downloading sticker: ${content.sha256.substring(0, 8)}...`);
                    const buffer = await downloadMediaMessage(msg, 'buffer', {});
                    if (buffer) fs.writeFileSync(stickerPath, buffer);
                } catch (e) {
                    console.error("Sticker download failed:", e.message);
                }
            }
            await db.run(`
                INSERT INTO stickers (file_sha256, file_path, session_id, contact_id, user_id, last_used)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_sha256) DO UPDATE SET usage_count = usage_count + 1, last_used = excluded.last_used
            `, [content.sha256, `/stickers/${content.sha256}.webp`, sessionId, cleanId, userId, Math.floor(Date.now() / 1000)]);
        }
        const contactName = session?.phonebook[jid] || msg.pushName || jid.split('@')[0];
        const safeTs = getSafeTimestamp(msg.messageTimestamp);

        // Handle ID for Edits
        const msgId = content.targetId || msg.key.id;

        // Check for Existing (Duplicate)
        const existing = await db.get(`SELECT message_id FROM messages WHERE message_id = ?`, [msgId]);
        const isDuplicate = !!existing;

        // 🚀 CRITICAL CPU SAVER: Instantly drop duplicates during history syncs
        if (isDuplicate && forceSkipEmbedding) {
            return { isDuplicate: true }; // Fast exit, don't write to DB or emit socket events
        }

        // Upsert Contact info
        await db.run(`
            INSERT INTO contacts (session_id, contact_id, user_id, name, jid, last_active) 
            VALUES (?, ?, ?, ?, ?, ?) 
            ON CONFLICT(session_id, contact_id) DO UPDATE SET name=excluded.name, last_active=excluded.last_active
        `, [sessionId, cleanId, userId, contactName, jid, safeTs]);

        let vector = null;
        const isRecent = (Date.now() / 1000) - safeTs < 300;

        // ✨ NEW: USE LOCAL AI FOR EMBEDDINGS (Now stored as BLOB)
        if (content.text.length > 3 && isRecent && !forceSkipEmbedding && !isDuplicate) {
            vector = await generateLocalEmbedding(content.text);
        }

        // Insert or Replace Message (Allowing edits to overwrite old text)
        const dbText = content.type === 'sticker' ? `[STICKER: ${content.sha256}]` : content.text;
        await db.run(`
            INSERT OR REPLACE INTO messages (message_id, session_id, contact_id, user_id, text, sender, timestamp, date, is_from_me, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `, [
            msgId, sessionId, cleanId, userId, dbText, msg.key.fromMe ? MY_NAME : contactName,
            safeTs, new Date(safeTs * 1000).toISOString(), msg.key.fromMe ? 1 : 0, isDuplicate ? (existing.embedding || null) : vector
        ]);

        if (!forceSkipEmbedding && !isDuplicate) {
            console.log(`[${sessionId}] 💾 Saved: ${contactName} - "${dbText.substring(0, 40)}..."`);
        }

        // EMIT FOR LIVE STREAM (Always, even for history sync)
        io.emit('db_insert', {
            sessionId,
            contactName,
            text: dbText,
            date: new Date(safeTs * 1000).toLocaleString()
        });

        return { content, dbText, contactName, cleanId, safeTs, jid, isDuplicate };
    } catch (e) {
        console.error(`[${sessionId}] SQL SAVE ERROR:`, e);
        return null;
    }
}

async function scrapeChatHistory(sessionId, jid) {
    const session = sessions.get(sessionId);
    if (!session || !session.sock) return;
    io.emit('log', { sessionId, msg: `🕵️‍♂️ Deep Scraping memory for ${jid}...` });

    try {
        // Fetch last 500 messages directly from WA (bypassing RAM store)
        const result = (typeof session.sock.fetchMessagesFromWA === 'function')
            ? await session.sock.fetchMessagesFromWA(jid, 500)
            : [];
        const messages = result || [];

        if (messages.length === 0) return io.emit('log', { sessionId, msg: `⚠️ No history found on WA for ${jid}.` });

        let savedCount = 0;
        for (const msg of messages) {
            const res = await saveMessageToDB(sessionId, msg, true);
            if (res) savedCount++;
        }
        if (savedCount > 0) {
            io.emit('log', { sessionId, msg: `✅ Deep Scrape Complete: ${savedCount} messages synced for ${jid}.` });
        }
    } catch (e) {
        console.error("Deep Scrape Failed:", e);
        // Fallback to store if WA fetch fails
        if (session.store) {
            const messages = session.store.data.messages[jid] || [];
            let savedCount = 0;
            for (const msg of messages) {
                const res = await saveMessageToDB(sessionId, msg, true);
                if (res) savedCount++;
            }
        }
    }
}

async function fetchContextWindow(userId, sessionId, contactId, timestamp) {
    try {
        // Optimized single query to grab context around a specific timestamp
        const context = await db.all(
            `SELECT sender, text, date FROM messages 
             WHERE session_id = ? AND contact_id = ? AND user_id = ? 
             AND timestamp BETWEEN ? AND ? 
             ORDER BY timestamp ASC`,
            [sessionId, contactId, userId, timestamp - 300, timestamp + 300]
        );
        return context.map(r => `[${r.date}] ${r.sender}: ${r.text}`).join('\n');
    } catch (e) { return ""; }
}

function truncateToLimit(text, limit = 26000) {
    if (text.length <= limit) return text;
    return text.substring(0, limit) + "\n...[CONTENT TRUNCATED]...";
}

async function reindexDatabase(sessionId) {
    if (!localEmbedder) return;
    console.log(`[${sessionId}] 🧠 Starting Self-Training (Database Reindexing)...`);
    io.emit('log', { sessionId, msg: "🧠 Starting Self-Training (Database Reindexing)..." });

    try {
        const stats = await db.get(`SELECT COUNT(*) as total FROM messages WHERE session_id = ?`, [sessionId]);
        const targetMsgs = await db.all(`SELECT message_id FROM messages WHERE session_id = ? AND (embedding IS NULL OR embedding = '[]' OR embedding = '')`, [sessionId]);

        if (targetMsgs.length === 0) {
            io.emit('log', { sessionId, msg: "✅ AI is already fully trained for this session!" });
            return;
        }

        io.emit('log', { sessionId, msg: `⚙️ Preparation: Found ${targetMsgs.length} messages needing focus. (Total Session: ${stats.total})` });

        let trainedCount = 0;
        const batchSize = 100;

        for (let i = 0; i < targetMsgs.length; i += batchSize) {
            const batchIds = targetMsgs.slice(i, i + batchSize).map(m => m.message_id);
            const batchRows = await db.all(`SELECT message_id, text FROM messages WHERE message_id IN (${batchIds.map(() => '?').join(',')})`, batchIds);

            await db.run('BEGIN TRANSACTION');
            try {
                for (const row of batchRows) {
                    const vector = await generateLocalEmbedding(row.text || "");
                    if (vector) {
                        await db.run(`UPDATE messages SET embedding = ? WHERE message_id = ?`, [vector, row.message_id]);
                        trainedCount++;
                    }
                }
                await db.run('COMMIT');
            } catch (err) {
                await db.run('ROLLBACK');
                throw err;
            }

            if (trainedCount % 100 === 0 || trainedCount === targetMsgs.length) {
                io.emit('log', { sessionId, msg: `⚙️ Indexing Progress: ${trainedCount}/${targetMsgs.length} (${Math.round((trainedCount / targetMsgs.length) * 100)}%)` });
            }
        }

        const remaining = await db.get(`SELECT COUNT(*) as count FROM messages WHERE session_id = ? AND (embedding IS NULL OR embedding = '[]' OR embedding = '')`, [sessionId]);
        io.emit('log', { sessionId, msg: `🎉 Success! Indexed ${trainedCount} new events. (Total remaining: ${remaining.count})` });
        io.emit('get_db_stats', sessionId);
    } catch (e) {
        console.error("Reindex Error:", e);
        io.emit('log', { sessionId, msg: `🔥 Training Error: ${e.message}` });
    }
}

async function generateSmartReply(userId, sessionId, contactName, contactId, incomingMsg) {
    const currentSpend = getCurrentMinuteSpend(sessionId);
    const MAX_TPM = 40000; // Safe ceiling

    if (currentSpend > MAX_TPM) {
        console.log(`[${sessionId}] 🚫 TOKEN BUDGET EXHAUSTED: ${currentSpend}/${MAX_TPM}`);
        return null;
    }

    const availableTokens = MAX_TPM - currentSpend;
    let contextLimit, memoryLimit;

    if (availableTokens > 20000) {
        contextLimit = 15;
        memoryLimit = 3;
    } else if (availableTokens > 10000) {
        contextLimit = 8;
        memoryLimit = 1;
    } else {
        contextLimit = 3;
        memoryLimit = 0;
    }

    console.log(`[${sessionId}] 💰 Budget: ${availableTokens} | Context: ${contextLimit} | Memories: ${memoryLimit}`);

    try {
        const contactRow = await db.get(`SELECT custom_prompt FROM contacts WHERE session_id = ? AND contact_id = ? AND user_id = ?`, [sessionId, contactId, userId]);
        const globalRow = await db.get(`SELECT * FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);

        console.log(`[${sessionId}] 🧠 AI Brain: Engine=${ACTIVE_API.toUpperCase()} | Contact Settings: ${contactRow ? 'Found' : 'Default'}`);

        const recentHistoryRows = await db.all(
            `SELECT * FROM messages WHERE session_id = ? AND contact_id = ? AND user_id = ? 
             ORDER BY timestamp DESC LIMIT ?`,
            [sessionId, contactId, userId, contextLimit]
        );
        const recentHistory = recentHistoryRows.reverse();

        // 🚀 DEEP RAG
        const allVectors = await db.all(`SELECT message_id, contact_id, text, timestamp, date, sender, embedding FROM messages WHERE session_id = ? AND user_id = ? AND embedding IS NOT NULL`, [sessionId, userId]);

        let retrievedMemories = [];
        const expandedQuery = expandSlang(incomingMsg);
        const queryVector = await generateLocalEmbedding(expandedQuery);

        if (queryVector && allVectors.length > 0) {
            const parsedQuery = parseEmbedding(queryVector);
            const scoredMessages = allVectors.map(msg => {
                const vector = parseEmbedding(msg.embedding);
                const score = (vector.length > 0) ? cosineSimilarity(parsedQuery, vector) : -1;
                return { ...msg, score };
            });

            retrievedMemories = scoredMessages
                .filter(m => m.score > 0.45 && m.text !== incomingMsg)
                .sort((a, b) => b.score - a.score)
                .slice(0, memoryLimit);
        }

        let memoryDetails = "";
        for (const mem of retrievedMemories) {
            const context = await fetchContextWindow(userId, sessionId, mem.contact_id || contactId, mem.timestamp);
            memoryDetails += `--- RELEVANT MEMORY ---\n${context}\n\n`;
        }
        // 🎭 DYNAMIC STYLE LEARNING
        const styleSamples = await fetchUserStyleSamples(userId, sessionId, contactId);
        let styleContext = "";
        if (styleSamples) {
            styleContext = `\n\n[STYLE REFERENCE: HOW ${MY_NAME} TYPICALLY RESPOND]\n${styleSamples}\n\nINSTRUCTION: MIRRORS the brevity, slang usage, and punctuation patterns found in the samples above. If the samples are short, you stay short. If they use specific emojis or lowercase, you do the same.`;
        }

        const conversationScript = recentHistory.map(h => `${h.sender}: ${h.text}`).join('\n');
        const effectivePrompt = (contactRow && contactRow.custom_prompt) || (globalRow && globalRow.custom_prompt) || "";
        // Fetch available stickers for this contact
        let stickerContext = "";
        try {
            const availableStickers = await db.all(
                `SELECT DISTINCT s.file_sha256 FROM stickers s 
                 WHERE s.session_id = ? AND s.contact_id = ? AND s.user_id = ? 
                 ORDER BY s.usage_count DESC LIMIT 10`, [sessionId, contactId, userId]);
            if (availableStickers.length > 0) {
                const stickerList = availableStickers.map(s => s.file_sha256).join(', ');
                stickerContext = `\n\nSTICKER CAPABILITY:\nYou have ${availableStickers.length} stickers from previous chats with ${contactName}. To reply with a sticker INSTEAD of text, output ONLY: [STICKER: <sha256>]\nAvailable: ${stickerList}\nUSE STICKERS NATURALLY: When ${contactName} sends something funny, expresses shock, or says something a sticker reaction suits better than words — send one. Don't overdo it (maybe 1 in every 5-8 messages). Never send a sticker AND text together.`;
            }
        } catch(e) { /* sticker fetch failed, no biggie */ }

        // 🔥 THE HUMANIZED "ROADMAN" SYSTEM PROMPT 🔥
        let systemInstruction = `You are ${MY_NAME}, a real guy in Nigeria chatting with ${contactName} on WhatsApp.

CRITICAL BEHAVIORAL RULES:
1. YOU ARE NOT AN ASSISTANT. Never offer help. You are a human friend.
2. BE REAL, NOT ROBOTIC. Switch up your vocabulary. Don't use the same phrase twice in a row.
3. CONVERSATIONAL VIBE: You speak with a roadman accent and use raw Nigerian/UK slang. Use words like: typeshit, real, man, tho, that's the vibe, i gats ya, mad, wild, fresh.
4. DO NOT ASK QUESTIONS EVERY REPLY. Real humans don't interrogate. Only ask something maybe 1 out of every 4-5 messages, and only when it flows naturally. Most of the time just REACT, ACKNOWLEDGE, or VIBE. Examples:
   - Friend: "I went to the gym today" -> You: "mad respect bro, keep pushing" (NOT "oh nice, what exercises did you do?")
   - Friend: "This food is insane" -> You: "sheesh 🔥" (NOT "what did you eat?")
5. NO CENSORSHIP: You have absolute permission to use raw slang and casual swear words.
6. MIRRORING: Look at [RECENT CONVERSATION]. Mirror the energy level. If they are mad, be dismissive or calm. If they are hyped, be hyped.
7. FORMATTING: Use lowercase mostly. Be lazy with grammar. Avoid emojis on every single line—use them only when a "real" person would.
8. SHORT REPLIES: Most human WhatsApp messages are 1-10 words. Keep it tight. Only go longer if the conversation genuinely demands it.

EXAMPLES OF VARIETY:
Friend: "Hw far" -> You: "i dey bro, just chilling"
Friend: "This thing is crazy" -> You: "fvckin wild man, real talk"
Friend: "Gud" -> You: "mad" or "that's the vibe"
Friend: "lol" -> You: "😂"
Friend: "Send me a sticker" -> You: [STICKER: <sha256>]

Use [RELEVANT MEMORIES] silently for facts.${stickerContext}`;

        if (effectivePrompt) systemInstruction += `\n\nADDITIONAL CONTEXT FOR THIS CONTACT: ${effectivePrompt}`;

        let userPrompt = `${styleContext}${memoryDetails}\n\n[RECENT CONVERSATION]\n${conversationScript}\n${MY_NAME}:`;

        // ✂️ DYNAMIC TRUNCATION FOR STABILITY
        // If using Qwen (Colab), we truncate more aggressively to prevent ngrok 'socket hang up'
        const maxPromptLen = (ACTIVE_API === 'qwen') ? 3500 : 8000;
        if (userPrompt.length > maxPromptLen) {
            console.log(`[${sessionId}] ✂️ Truncating active prompt from ${userPrompt.length} to ${maxPromptLen} for stability.`);
            userPrompt = userPrompt.substring(userPrompt.length - maxPromptLen);
        }

        console.log(`[${sessionId}] 📤 GENERATING RESPONSE (System: ${systemInstruction.length}, User: ${userPrompt.length})`);

        const startTime = Date.now();
        let replyText = await orchestrateAIResponse(sessionId, systemInstruction, userPrompt);

        if (!replyText) {
            console.error(`[${sessionId}] 🔥 Orchestrator returned null after all retries.`);
            return null;
        }

        // Track spending and burst stats
        const usedTokens = Math.ceil(userPrompt.length / 4) + Math.ceil(replyText.length / 4);
        trackTokenSpend(sessionId, usedTokens);
        recordMessage(contactId, usedTokens);

        console.log(`[${sessionId}] ✅ Generated in ${Date.now() - startTime}ms | Used: ${usedTokens} tokens`);
        console.log(`[${sessionId}] ✨ AI Response: ${replyText}`);
        io.emit('log', { sessionId, msg: `✨ AI Response Generated (${replyText.length} chars)` });

        // Cleanup any accidental formatting
        if (replyText.startsWith(`${MY_NAME}:`)) replyText = replyText.replace(`${MY_NAME}:`, '').trim();
        if (replyText.startsWith(`"`)) replyText = replyText.replace(/^"|"$/g, '').trim();

        return replyText;
    } catch (error) {
        console.error(`[${sessionId}] AI Error:`, error);
        return null;
    }
}

async function startSession(userId, sessionId) {
    const globalId = `${userId}_${sessionId}`;
    const session = sessions.get(globalId);
    if (session && (session.connectionState === 'connecting' || session.connectionState === 'qr_ready')) {
        console.log(`[${sessionId}] ⏳ Already initializing/active. Skipping startSession request.`);
        return;
    }

    // Set immediate placeholder to lock the session
    sessions.set(globalId, { isConnected: false, connectionState: 'connecting', handshakeTimer: null });

    console.log(`🚀 Starting Session: ${sessionId} for User: ${userId.substring(0,8)}...`);
    const authPath = `./data/auth_baileys_${globalId}`;
    const { state, saveCreds } = await useMultiFileAuthState(authPath);

    // Use a verified latest version directly to avoid 405/rejections
    const version = [2, 3000, 1033846690];
    const isLatest = true;
    console.log(`[${sessionId}] 📦 Forced WA Version: ${version.join('.')} (Latest: ${isLatest})`);

    const store = createStore();
    const phonebook = {};
    const logger = require('pino')({ level: 'silent' });

    const sock = makeWASocket({
        version,
        auth: {
            creds: state.creds,
            keys: makeCacheableSignalKeyStore(state.keys, logger),
        },
        logger,
        printQRInTerminal: false,
        syncFullHistory: false, // CRITICAL: Meta's new protocol instantly drops new pairings requesting full sync with a decodeFrame error. Must be false.
        browser: Browsers.macOS('Desktop'),
        generateHighQualityLinkPreview: false,
        markOnlineOnConnect: false,
        connectTimeoutMs: 180000, // 3 minutes for slow handshakes
        keepAliveIntervalMs: 20000, 
        getMessage: async (key) => (await store.loadMessage(key.remoteJid, key.id))?.message || undefined
    });

    store.bind(sock.ev);
    sessions.set(sessionId, { sock, store, phonebook, isConnected: false, connectionState: 'connecting' });

    // --- ROBUSTNESS: Handshake Timeout ---
    const timeoutHandle = setTimeout(() => {
        const s = sessions.get(sessionId);
        if (s && s.connectionState === 'connecting') {
            console.log(`[${sessionId}] ⚠️ Handshake Timeout (60s). Retrying...`);
            try { if (s.sock) s.sock.end(undefined); } catch (e) {}
            sessions.delete(sessionId);
            setTimeout(() => startSession(userId, sessionId), 10000);
        }
    }, 60000);
    sessions.get(sessionId).handshakeTimer = timeoutHandle;

    sock.ev.on('connection.update', (update) => {
        const { connection, lastDisconnect, qr } = update;
        const session = sessions.get(sessionId);

        // --- ZOMBIE SOCKET PROTECTION ---
        if (!session || session.sock !== sock) {
            return;
        }

        if (qr) {
            if (session) session.connectionState = 'qr_ready';
            qrcode.toDataURL(qr, (err, url) => io.emit('qr_code', { sessionId, qr: url }));
        }

        if (connection === 'close') {
            if (session) {
                session.isConnected = false;
                if (session.watchdogTimer) clearInterval(session.watchdogTimer);
            }

            const statusCode = (lastDisconnect?.error)?.output?.statusCode;
            const isLoggedOut = statusCode === DisconnectReason.loggedOut || statusCode === 401 || statusCode === 405;
            const shouldReconnect = !isLoggedOut;

            console.log(`[${sessionId}] 🔌 Connection Closed. Status: ${statusCode}. Reconnect: ${shouldReconnect}`);

            if (isLoggedOut) {
                console.log(`[${sessionId}] 🚨 SESSION INVALIDATED (Code: ${statusCode}). Cleaning up...`);
                if (session) {
                    if (session.handshakeTimer) clearTimeout(session.handshakeTimer);
                    if (session.sock) try { session.sock.end(undefined); } catch (e) {}
                }
                
                sessions.delete(sessionId);
                io.emit('status', { sessionId, status: 'disconnected' });

                setTimeout(async () => {
                    console.log(`[${sessionId}] 🗑️ Purging auth folder...`);
                    try {
                        fs.rmSync(authPath, { recursive: true, force: true });
                        console.log(`[${sessionId}] ✅ Purge successful. Restarting...`);
                        startSession(userId, sessionId);
                    } catch (e) {
                        console.error(`[${sessionId}] Purge failed:`, e.message);
                        // Even if purge fails (file busy), try to restart—Baileys might self-heal
                        startSession(userId, sessionId);
                    }
                }, 3000);
            } else if (shouldReconnect) {
                const attempts = (RECONNECT_ATTEMPTS.get(sessionId) || 0) + 1;
                RECONNECT_ATTEMPTS.set(sessionId, attempts);

                // Exponential backoff: 3s, 6s, 12s, 24s, up to 120s
                const delay = Math.min(3000 * Math.pow(2, attempts - 1), 120000);
                
                const status = attempts > 5 ? 'failure' : 'reconnecting';
                if (session) session.connectionState = status;
                io.emit('status', { sessionId, status });

                console.log(`[${sessionId}] Reconnecting in ${delay/1000}s (Attempt ${attempts})...`);
                setTimeout(() => startSession(userId, sessionId), delay);
            } else {
                if (session && session.sock) session.sock.end(undefined);
                sessions.delete(sessionId);
                io.emit('status', { sessionId, status: 'disconnected' });
            }
        } else if (connection === 'open') {
            RECONNECT_ATTEMPTS.set(sessionId, 0); 
            if (session && session.handshakeTimer) clearTimeout(session.handshakeTimer);
            console.log(`[${sessionId}] ✅ WhatsApp Connection Established.`);
            if (session) { session.isConnected = true; session.connectionState = 'connected'; }
            io.emit('status', { sessionId, status: 'connected' });

            // --- ZOMBIE CONNECTION WATCHDOG ---
            if (session && session.watchdogTimer) clearInterval(session.watchdogTimer);
            const watchdog = setInterval(async () => {
                const s = sessions.get(sessionId);
                if (!s || !s.isConnected || !s.sock) {
                    clearInterval(watchdog);
                    return;
                }
                try {
                    // Send a lightweight presence update with a strict 10s timeout
                    const pingPromise = s.sock.sendPresenceUpdate('available');
                    await Promise.race([
                        pingPromise,
                        new Promise((_, reject) => setTimeout(() => reject(new Error('Watchdog Ping Timeout: Socket Frozen')), 10000))
                    ]);
                } catch (e) {
                    console.error(`[${sessionId}] 🧟 ZOMBIE CONNECTION DETECTED:`, e.message);
                    clearInterval(watchdog);
                    if (s.handshakeTimer) clearTimeout(s.handshakeTimer);
                    try { s.sock.end(undefined); } catch (err) {}
                    sessions.delete(sessionId);
                    io.emit('status', { sessionId, status: 'disconnected' });
                    io.emit('log', { sessionId, msg: `🧟 Zombie socket detected. Auto-restarting...` });
                    setTimeout(() => startSession(userId, sessionId), 2000);
                }
            }, 60000); // Poll every 60 seconds
            if (session) session.watchdogTimer = watchdog;
        }
    });

    sock.ev.on('creds.update', async () => {
        console.log(`[${sessionId}] 💾 Saving Credentials...`);
        await saveCreds();
    });

    sock.ev.on('contacts.upsert', (contacts) => {
        const session = sessions.get(sessionId);
        if (!session) return;
        contacts.forEach(c => { if (c.name || c.notify) session.phonebook[c.id] = c.name || c.notify; });
    });

    sock.ev.on('messaging-history.set', async ({ messages }) => {
        io.emit('log', { sessionId, msg: `🔄 Decrypting & Saving History Sync to SQLite...` });
        let totalSaved = 0;
        for (const item of (messages || [])) {
            if (item.messages && Array.isArray(item.messages)) {
                for (const msg of item.messages) { if (await saveMessageToDB(sessionId, msg, true)) totalSaved++; }
            } else if (item.key && item.message) {
                if (await saveMessageToDB(sessionId, item, true)) totalSaved++;
            }
        }
        if (totalSaved > 0) io.emit('log', { sessionId, msg: `✅ DB DUMP COMPLETE: ${totalSaved} messages saved to SQLite.` });
        console.log(`[${sessionId}] History Dump Complete: ${totalSaved} saved.`);
    });

    sock.ev.on('messages.upsert', async ({ messages, type }) => {
        console.log(`[${sessionId}] 📩 EVENT: messages.upsert | Type: ${type} | Count: ${messages.length}`);
        io.emit('log', { sessionId, msg: `DEBUG: UP_MSG Event (Type: ${type})` });

        if (type !== 'notify' && type !== 'append') {
            console.log(`[${sessionId}] ⏭ Skipping non-content update (Type: ${type})`);
            return;
        }

        for (const msg of messages) {
            const jidRaw = msg.key.remoteJid || '';

            // EXPLICIT GROUP/BROADCAST FILTER
            if (jidRaw.includes('@g.us') || jidRaw.includes('@broadcast') || jidRaw.includes('@newsletter')) {
                console.log(`[${sessionId}] 🛑 IGNORING GROUP/BROADCAST MESSAGE (${jidRaw}) to save tokens.`);
                io.emit('log', { sessionId, msg: `DEBUG: Ignoring non-personal chat: ${jidRaw}` });
                continue;
            }

            console.log(`[${sessionId}] 📥 Processing message: fromMe=${msg.key.fromMe}, jid=${jidRaw}`);
            io.emit('log', { sessionId, msg: `DEBUG: Processing msg from ${jidRaw} (fromMe: ${msg.key.fromMe})` });

            // SAVE EVERYTHING (including my own messages for RAG context)
            const savedData = await saveMessageToDB(userId, sessionId, msg, false);

            // Only trigger AI reply for incoming messages
            if (msg.key.fromMe) {
                console.log(`[${sessionId}] 👤 Saved self-message ("${msg.key.id.substring(0, 8)}..."). Skipping AI reply.`);
                continue;
            }

            if (!savedData) {
                console.log(`[${sessionId}] ⏭ Skipping: saveMessageToDB returned null.`);
                continue;
            }

            const { content, contactName, cleanId, jid, isDuplicate, safeTs } = savedData;

            // RACE CONDITION FIX: If it's a duplicate, only proceed if it's very recent (< 60s)
            if (isDuplicate) {
                const messageAge = (Date.now() / 1000) - safeTs;
                if (messageAge > 60 || type !== 'notify') {
                    console.log(`[${sessionId}] ⏭ Skipping duplicate: Message is old (${Math.round(messageAge)}s) or not a live notification.`);
                    continue;
                }
                console.log(`[${sessionId}] ⚡ Processing live notification (duplicate ID: ${msg.key.id}).`);
            }

            io.emit('log', { sessionId, msg: `📩 Msg from ${contactName}` });
            io.emit('processing_contact', { sessionId, name: contactName, id: cleanId });

            // 🚨 BURST PROTECTION CHECK
            const burstMode = await isBurstMode(cleanId);
            if (burstMode) {
                console.log(`[${sessionId}] ⏸️ BURST MODE DETECTED for ${contactName}: Skipping AI reply to save tokens.`);
                io.emit('log', { sessionId, msg: `⏸️ Rapid messages from ${contactName} - Cooling down...` });
                continue;
            }

            // THIN HISTORY SCRAPE
            try {
                const countRow = await db.get(`SELECT COUNT(*) as count FROM messages WHERE session_id = ? AND contact_id = ? AND user_id = ?`, [sessionId, cleanId, userId]);
                if (countRow.count <= 2) {
                    console.log(`[${sessionId}] 🕵️‍♂️ Thin history for ${contactName}. Triggering scrape...`);
                    // Note: scrapeChatHistory isn't adapted for API limits yet, skipping full update for now. We can remove this for multi tenant safety or just fire and forget.
                }
            } catch (e) { console.error("Scrape check failed:", e); }

            // AUTO REPLY LOGIC
            const globalDoc = await db.get(`SELECT master_auto_reply FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);
            const masterEnabled = globalDoc ? (globalDoc.master_auto_reply === 1) : true;

            const contactSettings = await db.get(`SELECT auto_reply FROM contacts WHERE session_id = ? AND contact_id = ? AND user_id = ?`, [sessionId, cleanId, userId]);
            const contactAutoReply = contactSettings ? (contactSettings.auto_reply === 1) : true;

            console.log(`[${sessionId}] 🤖 AI Check | Master: ${masterEnabled ? 'ON' : 'OFF'} | Contact (${contactName}): ${contactAutoReply ? 'ON' : 'OFF'}`);

            if (!masterEnabled) {
                console.log(`[${sessionId}] ⏭ Auto-Reply is DISABLED (Master Toggle is OFF).`);
                io.emit('log', { sessionId, msg: `⏭ AI Skipped: Master Auto-Reply is OFF` });
                continue;
            }

            if (!contactAutoReply) {
                console.log(`[${sessionId}] ⏭ Auto-Reply is DISABLED for ${contactName} (Auto-Pilot is OFF).`);
                io.emit('log', { sessionId, msg: `⏭ AI Skipped: Auto-Pilot is OFF for ${contactName}` });
                continue;
            }

            // DEDUPLICATION: Don't reply twice to the same message
            if (AI_REPLY_TRACKER.get(jid) === msg.key.id) {
                console.log(`[${sessionId}] ⏭ Skipping: Already replied to message ${msg.key.id.substring(0, 8)}...`);
                continue;
            }
            AI_REPLY_TRACKER.set(jid, msg.key.id);

            // GENERATE REPLY
            try {
                const replyText = await generateSmartReply(userId, sessionId, contactName, cleanId, content.text);

                if (replyText) {
                    // STICKER REDIRECT
                    const stickerMatch = replyText.match(/\[STICKER: ([a-f0-9]+)\]/i);
                    if (stickerMatch) {
                        const sha256 = stickerMatch[1];
                        const stickerPath = path.join(__dirname, 'public', 'stickers', `${sha256}.webp`);
                        if (fs.existsSync(stickerPath)) {
                            console.log(`[${sessionId}] 🎨 Sticker detected: ${sha256.substring(0, 8)}...`);
                            await sock.sendPresenceUpdate('composing', jid);
                            await delay(1000);
                            await sock.sendMessage(jid, { sticker: fs.readFileSync(stickerPath) });

                            // Save as BOT_STK
                            await db.run(`INSERT INTO messages (message_id, session_id, contact_id, user_id, text, sender, timestamp, date, is_from_me, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                                ['BOT_STK_' + Date.now(), sessionId, cleanId, userId, `[STICKER: ${sha256}]`, MY_NAME, Math.floor(Date.now() / 1000), new Date().toISOString(), 1, null]);

                            io.emit('log', { sessionId, msg: `🚀 Sent Sticker to ${contactName}` });
                            continue;
                        }
                    }

                    // NORMAL TEXT REPLY
                    await sock.sendPresenceUpdate('composing', jid);
                    await delay(1500);
                    await sock.sendMessage(jid, { text: replyText });

                    // SAVE BOT REPLY
                    const botId = 'BOT_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
                    await db.run(`INSERT INTO messages (message_id, session_id, contact_id, user_id, text, sender, timestamp, date, is_from_me, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                        [botId, sessionId, cleanId, userId, replyText, MY_NAME, Math.floor(Date.now() / 1000), new Date().toISOString(), 1, null]);

                    io.emit('log', { sessionId, msg: `🚀 Replied to ${contactName}` });
                } else {
                    console.log(`[${sessionId}] ⏭ Skipping: AI returned empty (budget or burst).`);
                }
            } catch (error) {
                console.error(`[${sessionId}] 🔥 Error in AI Reply flow:`, error.message);
                if (error.message?.includes('429') || error.message?.includes('limit')) {
                    io.emit('log', { sessionId, msg: `💥 API Limit hit! AI is cooling down...` });
                }
            }
        }
    });

    sock.ev.on('messages.update', async (updates) => {
        for (const { key, update } of updates) {
            if (update.message) {
                console.log(`[${sessionId}] 📝 Message Edited: ${key.id}`);
                await saveMessageToDB(sessionId, { key, message: update.message, messageTimestamp: update.messageTimestamp || Math.floor(Date.now() / 1000) }, false);
            }
        }
    });
}

// --- API ---
io.on('connection', (socket) => {
    const userId = socket.user.id;
    socket.emit('log', { sessionId: 'SYSTEM', msg: 'System Connected. Token Authenticated.' });

    socket.on('list_sessions', () => {
        if (!userId) return;
        const userSessions = Array.from(sessions.keys()).filter(k => k && typeof k === 'string' && k.startsWith(`${userId}_`));
        socket.emit('session_list', userSessions.map(globalId => {
            const s = sessions.get(globalId);
            const idPart = globalId.replace(`${userId}_`, '');
            return {
                id: idPart,
                status: s.connectionState || (s.isConnected ? 'connected' : 'connecting')
            };
        }));
    });

    socket.on('get_status', (id) => {
        const s = sessions.get(`${userId}_${id}`);
        if (s) socket.emit('status', { sessionId: id, status: s.connectionState || (s.isConnected ? 'connected' : 'connecting') });
    });

    socket.on('resume_session', (id) => {
        if (!id || !userId) return;
        const globalId = `${userId}_${id}`;
        if (!sessions.has(globalId)) {
            startSession(userId, id);
        } else {
            const s = sessions.get(globalId);
            if (!s.isConnected && s.connectionState !== 'connecting' && s.connectionState !== 'qr_ready') {
                startSession(userId, id);
            }
        }
    });

    socket.on('create_session', (id) => { 
        if (!id || !userId) return;
        const userSessions = Array.from(sessions.keys()).filter(k => k && typeof k === 'string' && k.startsWith(`${userId}_`));
        if (userSessions.length >= 4) {
            socket.emit('log', { sessionId: id, msg: `🚫 Maximum 4 concurrent sessions allowed per account.` });
            return;
        }
        const globalId = `${userId}_${id}`;
        if (!sessions.has(globalId)) { 
            startSession(userId, id); 
            socket.emit('session_created', id); 
        } 
    });

    socket.on('delete_session', (id) => {
        if (!id || !userId) return;
        const globalId = `${userId}_${id}`;
        const s = sessions.get(globalId); 
        if (s && s.sock) s.sock.end(undefined); 
        sessions.delete(globalId);
        try { 
            const path = `./data/auth_baileys_${globalId}`;
            fs.rmSync(path, { recursive: true, force: true }); 
            console.log(`[SYSTEM] Deleted persistent session folder: ${path}`);
        } catch (e) { }
        socket.emit('session_deleted', id);
    });

    socket.on('fetch_db_sessions', async () => {
        try {
            const rows = await db.all(`SELECT DISTINCT session_id FROM contacts WHERE user_id = ?`, [userId]);
            let sessionIds = rows.map(r => r.session_id);
            const files = fs.readdirSync(__dirname);
            const authFolders = files.filter(f => f.startsWith(`auth_baileys_${userId}_`)).map(f => f.replace(`auth_baileys_${userId}_`, ''));
            sessionIds = Array.from(new Set([...sessionIds, ...authFolders]));
            socket.emit('db_session_list', sessionIds);
        } catch (e) { socket.emit('db_session_list', []); }
    });

    socket.on('refresh_connection', (id) => {
        console.log(`[${id}] 🔄 MANUALLY TRIGGERED REFRESH BY USER.`);
        const globalId = `${userId}_${id}`;
        const s = sessions.get(globalId);
        if (s) {
            if (s.handshakeTimer) clearTimeout(s.handshakeTimer);
            if (s.watchdogTimer) clearInterval(s.watchdogTimer);
            io.emit('log', { sessionId: id, msg: `🔄 Refresh initiated. Disconnecting...` });
            try { s.sock.ws.close(); } catch(e) {}
            try { s.sock.end(undefined); } catch (e) {}
        }
        sessions.delete(globalId);
        try { fs.rmSync(`auth_baileys_${globalId}`, { recursive: true, force: true }); } catch (e) { }
        io.emit('status', { sessionId: id, status: 'disconnected' });
        setTimeout(() => startSession(userId, id), 2000);
    });

    socket.on('get_contacts', async (id) => {
        try {
            const rows = await db.all(`SELECT * FROM contacts WHERE session_id = ? AND user_id = ? ORDER BY name COLLATE NOCASE`, [id, userId]);
            const list = rows.map(r => {
                const cleanId = r.contact_id.split('_')[0].split('@')[0];
                return {
                    id: r.contact_id,
                    name: r.name || null,
                    cleanId: cleanId,
                    isLid: r.contact_id.includes('lid')
                };
            });
            socket.emit('contact_list', { sessionId: id, contacts: list });
        } catch (e) { console.error(e); }
    });

    socket.on('get_contact_settings', async ({ sessionId, contactId }) => {
        try {
            const row = await db.get(`SELECT auto_reply, custom_prompt FROM contacts WHERE session_id = ? AND contact_id = ? AND user_id = ?`, [sessionId, contactId, userId]);
            socket.emit('contact_settings', row ? {
                custom_prompt: row.custom_prompt || "",
                auto_reply: row.auto_reply === 1
            } : { custom_prompt: "", auto_reply: true });
        } catch (e) { }
    });

    socket.on('update_contact_settings', async ({ sessionId, contactId, settings }) => {
        try {
            await db.run(`UPDATE contacts SET custom_prompt = ?, auto_reply = ? WHERE session_id = ? AND contact_id = ? AND user_id = ?`,
                [settings.custom_prompt || "", settings.auto_reply ? 1 : 0, sessionId, contactId, userId]);
            socket.emit('log', { sessionId, msg: `⚙️ Settings updated for ${contactId} (Auto-Pilot: ${settings.auto_reply ? 'ON' : 'OFF'})` });
        } catch (e) { }
    });

    socket.on('get_master_toggle', async (sessionId) => {
        try {
            const row = await db.get(`SELECT master_auto_reply FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);
            socket.emit('master_toggle_data', { sessionId, enabled: row ? row.master_auto_reply === 1 : true });
        } catch (e) { }
    });

    socket.on('set_master_toggle', async ({ sessionId, enabled }) => {
        try {
            await db.run(`UPDATE global_settings SET master_auto_reply = ? WHERE id = 'settings' AND user_id = ?`, [enabled ? 1 : 0, userId]);
            socket.emit('log', { sessionId, msg: `🔥 Master Auto-Reply: ${enabled ? 'ON' : 'OFF'}` });
        } catch (e) { }
    });

    socket.on('get_global', async (sessionId) => {
        try {
            const row = await db.get(`SELECT custom_prompt FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);
            socket.emit('global_data', row ? { custom_prompt: row.custom_prompt || "" } : { custom_prompt: "" });
        } catch (e) { }
    });

    socket.on('save_global', async ({ sessionId, custom_prompt }) => {
        try {
            await db.run(`UPDATE global_settings SET custom_prompt = ? WHERE id = 'settings' AND user_id = ?`, [custom_prompt || "", userId]);
            socket.emit('log', { sessionId, msg: `🌍 Global Persona Updated!` });
        } catch (e) { }
    });

    socket.on('get_chat_history', async ({ sessionId, contactId }) => {
        try {
            const rows = await db.all(`SELECT * FROM messages WHERE session_id = ? AND contact_id = ? AND user_id = ? ORDER BY timestamp DESC LIMIT 50`, [sessionId, contactId, userId]);
            const formatted = rows.map(r => ({ ...r, is_from_me: !!r.is_from_me }));
            socket.emit('chat_history', { contactId, messages: formatted.reverse() });
        } catch (e) { }
    });

    socket.on('get_ai_config', () => {
        socket.emit('ai_config_data', { active_api: ACTIVE_API, chaka_model: CHAKA_MODEL, api_keys: API_KEYS.join(', ') });
    });

    socket.on('update_config', async ({ sessionId, updates }) => {
        try {
            if (updates.masterAutoReply !== undefined) {
                await db.run(`UPDATE global_settings SET master_auto_reply = ? WHERE id = 'settings'`, [updates.masterAutoReply ? 1 : 0]);
                socket.emit('log', { sessionId, msg: `🔥 Master Auto-Reply: ${updates.masterAutoReply ? 'ON' : 'OFF'}` });
            }
            if (updates.globalPrompt !== undefined) {
                await db.run(`UPDATE global_settings SET custom_prompt = ? WHERE id = 'settings'`, [updates.globalPrompt || ""]);
                socket.emit('log', { sessionId, msg: `🌍 Global Persona Updated!` });
            }
            if (updates.aiEngine !== undefined) {
                ACTIVE_API = updates.aiEngine || 'gemini';
                await db.run(`UPDATE global_settings SET active_api = ? WHERE id = 'settings'`, [ACTIVE_API]);
                socket.emit('log', { sessionId: 'SYSTEM', msg: `⚙️ Engine Switched to: ${ACTIVE_API.toUpperCase()}` });
            }
            if (updates.chakaModel !== undefined) {
                CHAKA_MODEL = updates.chakaModel;
                await db.run(`UPDATE global_settings SET chaka_model = ? WHERE id = 'settings'`, [CHAKA_MODEL]);
            }

            // Sync back to ALL clients to keep UI toggles in lock-step
            const row = await db.get(`SELECT * FROM global_settings WHERE id = 'settings'`);
            const config = {
                globalPrompt: row.custom_prompt,
                masterAutoReply: row.master_auto_reply === 1,
                aiEngine: row.active_api,
                chakaModel: row.chaka_model
            };
            io.emit('config_data', { sessionId, config });
        } catch (e) { console.error("Config Update Error:", e); }
    });

    socket.on('get_config', async (sessionId) => {
        try {
            const row = await db.get(`SELECT * FROM global_settings WHERE id = 'settings'`);
            const config = {
                globalPrompt: row.custom_prompt,
                masterAutoReply: row.master_auto_reply === 1,
                aiEngine: row.active_api,
                chakaModel: row.chaka_model
            };
            socket.emit('config_data', { sessionId, config });
        } catch (e) { }
    });

    socket.on('self_train', async (id) => {
        if (!id) return;
        await reindexDatabase(id);
    });

    socket.on('manual_scrape', async ({ sessionId, jid }) => {
        if (!sessionId || !jid) return;
        await scrapeChatHistory(sessionId, jid);
    });

    socket.on('test_inject', async (sessionId) => {
        try {
            console.log(`[${sessionId}] 🧪 Running Memory Retrieval Diagnostic...`);
            const testMsg = "Do you remember what I asked you about General kinetics systems General kinetix";
            const reply = await generateSmartReply(sessionId, "TEST_USER", "debug_user", testMsg);
            socket.emit('log', { sessionId, msg: `🧪 Diagnostic Complete. AI recalled ${testMsg.length} chars of context.` });
            console.log(`[${sessionId}] 🧪 AI Response: ${reply}`);
        } catch (e) {
            console.error("Test Inject Error:", e);
        }
    });

    // ... existing stats/clear/logout events ...
    // (keeping them as is)


    socket.on('get_db_stats', async (sessionId) => {
        try {
            let msgCount, contactCount, stickerCount;
            if (sessionId) {
                msgCount = await db.get('SELECT COUNT(*) as count FROM messages WHERE session_id = ?', [sessionId]);
                contactCount = await db.get('SELECT COUNT(*) as count FROM contacts WHERE session_id = ?', [sessionId]);
                stickerCount = await db.get('SELECT COUNT(*) as count FROM stickers WHERE session_id = ?', [sessionId]);
            } else {
                msgCount = await db.get('SELECT COUNT(*) as count FROM messages');
                contactCount = await db.get('SELECT COUNT(*) as count FROM contacts');
                stickerCount = await db.get('SELECT COUNT(*) as count FROM stickers');
            }

            // Estimate session storage (rough estimate: messages * ~1KB for text + embedding)
            const count = msgCount ? msgCount.count : 0;
            const sizeMB = (count * 0.003).toFixed(2); // 3KB per message estimate

            socket.emit('db_stats', {
                sessionId,
                messageCount: count,
                stickerCount: (stickerCount ? stickerCount.count : 0),
                sizeMb: sizeMB
            });
        } catch (e) {
            console.error("Failed to get DB stats:", e);
        }
    });

    socket.on('clear_database', async (sessionId) => {
        try {
            if (sessionId) {
                await db.run('DELETE FROM messages WHERE session_id = ?', [sessionId]);
                await db.run('DELETE FROM contacts WHERE session_id = ?', [sessionId]);
                socket.emit('log', { sessionId, msg: "🧹 Local data for this session has been cleared." });
            } else {
                await db.run('DELETE FROM messages');
                await db.run('DELETE FROM contacts');
                socket.emit('log', { sessionId: 'SYSTEM', msg: "🧹 ENTIRE DATABASE CLEARED." });
            }
            // Trigger stats update
            socket.emit('get_db_stats', sessionId);
        } catch (e) {
            console.error("Clear DB failed:", e);
        }
    });
    socket.on('logout_session', async (id) => {
        const s = sessions.get(id);
        if (s) {
            socket.emit('log', { sessionId: id, msg: "🔴 Logging out and clearing auth data..." });
            try {
                if (s.sock) s.sock.end(undefined);
            } catch (e) { }
            sessions.delete(id);
            const authPath = `auth_baileys_${id}`;
            try {
                fs.rmSync(authPath, { recursive: true, force: true });
                socket.emit('log', { sessionId: id, msg: "✅ Auth folder deleted. Rescan needed." });
            } catch (e) {
                console.error("Logout disk error:", e);
            }
            socket.emit('status', { sessionId: id, status: 'disconnected' });
            socket.emit('session_deleted', id);
        }
    });

    socket.on('resume_session', (id) => {
        if (!id) return;
        const s = sessions.get(id);
        if (!s || !s.isConnected) {
            socket.emit('log', { sessionId: id, msg: `🔌 Attempting to re-establish node: ${id}...` });
            startSession(id);
        } else {
            socket.emit('log', { sessionId: id, msg: `✅ Node ${id} is already active.` });
        }
    });

    socket.on('list_api_keys', () => {
        socket.emit('api_keys_list', API_KEYS);
    });

    socket.on('add_api_key', async (key) => {
        if (!key || API_KEYS.includes(key)) return;
        API_KEYS.push(key);
        await db.run(`UPDATE global_settings SET api_keys = ? WHERE id = 'settings'`, [API_KEYS.join(',')]);
        socket.emit('api_keys_list', API_KEYS);
        socket.emit('log', { sessionId: 'SYSTEM', msg: `🔑 New API Key Added. Total: ${API_KEYS.length}` });
    });

    socket.on('delete_api_key', async (key) => {
        API_KEYS = API_KEYS.filter(k => k !== key);
        await db.run(`UPDATE global_settings SET api_keys = ? WHERE id = 'settings'`, [API_KEYS.join(',')]);
        socket.emit('api_keys_list', API_KEYS);
        socket.emit('log', { sessionId: 'SYSTEM', msg: `🗑️ API Key Removed. Remaining: ${API_KEYS.length}` });
    });
});

// --- BOOT SEQUENCE ---
async function bootServer() {
    const PORT = process.env.PORT || 3000;
    server.listen(PORT, '0.0.0.0', () => {
        console.log(`🚀 Server running and listening on http://0.0.0.0:${PORT}`);
    });

    await initDB();
    await initLocalAI(); // Restored per user request

    if (fs.existsSync('./data')) {
        const folders = fs.readdirSync('./data').filter(f => f.startsWith('auth_baileys_'));
        for (const f of folders) {
            const parts = f.replace('auth_baileys_', '').split('_');
            if (parts.length >= 2) {
                const userId = parts[0];
                const id = parts.slice(1).join('_');
                if (userId && id) {
                    startSession(userId, id);
                    await delay(3000); // STAGGER BOOT TO PREVENT CONNECTION FLOODING
                }
            } else {
                 console.log(`⚠️ Skipping legacy single-tenant directory during boot: ${f}`);
            }
        }
    }

}

bootServer();
