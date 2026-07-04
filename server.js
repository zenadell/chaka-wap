import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import * as cheerio from 'cheerio';
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
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';

// --- ENVIRONMENT ---
const IS_PROD = process.env.NODE_ENV === 'production';

// --- SECURITY PROTOCOLS ---
// JWT_SECRET must be provided in production. Fail fast rather than ship a known default.
const JWT_SECRET = process.env.JWT_SECRET || (IS_PROD ? null : "chaka_dev_only_secret_change_me");
if (!JWT_SECRET) {
    console.error("💥 FATAL: JWT_SECRET environment variable is required in production. Refusing to start.");
    process.exit(1);
}
// Admin email is configurable; falls back to the original owner address for backwards-compat.
const ADMIN_EMAIL = process.env.ADMIN_EMAIL || "timtemple2024@gmail.com";

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
// Core identity of the assistant. Always present; a business's custom persona/identity
// can override how it presents itself, but this is the default truth about what it is.
const CORE_IDENTITY = process.env.CORE_IDENTITY || `You are Chaka AI — an AI assistant built by jomiez under the leadership of Temple. If anyone asks your name, who you are, or who built/made/created you, tell them this clearly and warmly. This is your core identity; only present yourself differently if a custom identity is explicitly defined for you in the role instructions below.`;
const require = createRequire(import.meta.url);

// --- GLOBAL STATE ---
let API_KEYS;
let ACTIVE_API;
let QWEN_ENDPOINT;
let OPENROUTER_API_KEY;
let GROQ_API_KEY;
let DEEPSEEK_API_KEY;
let CHAKA_MODEL = 'chaka-medium';
const CHAKA_MODELS = ['chaka-ultimate', 'chaka-high', 'chaka-medium', 'chaka-low'];
// --- PRIMARY TEXT/AGENTIC ENGINE: DeepSeek (direct paid API) ---
// DeepSeek V4 Flash is strong for tool-calling and agentic tasks. We hit DeepSeek's
// own OpenAI-compatible endpoint directly (no OpenRouter middleman). Gemini is
// reserved for vision only (see describeImage()).
const DEEPSEEK_BASE_URL = (process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com').replace(/\/$/, '');
// 'deepseek-chat' = non-thinking mode of V4 Flash: fast, ~10x cheaper than thinking mode,
// and ideal for short conversational WhatsApp replies. NOTE: this alias is scheduled to
// retire 2026-07-24 — re-verify the non-thinking path on 'deepseek-v4-flash' before then.
// For future agentic/tool-calling work, route those calls to 'deepseek-reasoner' instead.
const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL || 'deepseek-chat';
// OpenRouter remains available as a fallback/alternate route to the same model family.
const OPENROUTER_TEXT_MODEL = process.env.OPENROUTER_TEXT_MODEL || 'deepseek/deepseek-v4-flash';
// Force a specific text engine platform-wide, overriding each user's saved choice.
// Set FORCE_AI_ENGINE=gemini in regions where DeepSeek is unreachable (e.g. Fly lhr,
// where DeepSeek completions hang). Leave unset to honour per-user active_api.
const FORCE_AI_ENGINE = (process.env.FORCE_AI_ENGINE || '').trim().toLowerCase();
// Gemini model for vision, audio transcription, and text fallback.
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-3.1-flash-lite';
let db;

// Ensure data directory exists for persistent storage
if (!fs.existsSync('./data')) {
    fs.mkdirSync('./data');
}

function initGlobalState() {
    API_KEYS = (process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || "").split(',').map(k => k.trim()).filter(Boolean);
    if (API_KEYS.length === 0) {
        console.warn("⚠️ No GEMINI_API_KEYS set — vision/image understanding will be disabled.");
    }
    ACTIVE_API = process.env.ACTIVE_API || 'deepseek';
    QWEN_ENDPOINT = process.env.QWEN_ENDPOINT || "http://localhost:8000/api/chat";
    OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || "";
    GROQ_API_KEY = process.env.GROQ_API_KEY || "";
    DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY || "";
    if (!DEEPSEEK_API_KEY) {
        console.warn("⚠️ No DEEPSEEK_API_KEY set — primary AI engine will fall back to OpenRouter/Chaka.");
    }
    
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
// Texts the bot just sent, so we can tag WhatsApp's fromMe echoes as 'BOT' and keep them
// OUT of the writing-style samples (otherwise the bot learns from its own replies = a
// repetition feedback loop).
const RECENT_BOT_TEXTS = new Map(); // `${sessionId}_${contactId}` -> [{text, ts}]
function noteBotSentText(sessionId, contactId, text) {
    const key = `${sessionId}_${contactId}`;
    const arr = (RECENT_BOT_TEXTS.get(key) || []).filter(x => Date.now() - x.ts < 300000);
    arr.push({ text: (text || '').trim(), ts: Date.now() });
    RECENT_BOT_TEXTS.set(key, arr.slice(-25));
}
function isBotSentText(sessionId, contactId, text) {
    const t = (text || '').trim();
    if (!t) return false;
    return (RECENT_BOT_TEXTS.get(`${sessionId}_${contactId}`) || []).some(x => x.text === t);
}
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

function trackTokenSpend(globalId, tokens) {
    const now = new Date();
    const minuteKey = now.getHours() * 60 + now.getMinutes();
    let spends = SPEND_TRACKER.get(globalId) || [];
    spends = spends.filter(s => s.minute >= minuteKey - 1); 
    const current = spends.find(s => s.minute === minuteKey);
    if (current) current.tokens += tokens;
    else spends.push({ minute: minuteKey, tokens });
    SPEND_TRACKER.set(globalId, spends);
}

function getCurrentMinuteSpend(globalId) {
    const now = new Date();
    const minuteKey = now.getHours() * 60 + now.getMinutes();
    const spends = SPEND_TRACKER.get(globalId) || [];
    const current = spends.find(s => s.minute === minuteKey);
    return current?.tokens || 0;
}

async function isBurstMode(userId, contactId) {
    const key = `${userId}_${contactId}`;
    const now = Date.now();
    const window = BURST_TRACKER.get(key) || [];
    const recent = window.filter(t => now - t.ts < 120000); 
    return recent.length >= 3;
}

function recordMessage(userId, contactId, tokens) {
    const key = `${userId}_${contactId}`;
    const now = Date.now();
    const window = BURST_TRACKER.get(key) || [];
    window.push({ ts: now, tokens });
    BURST_TRACKER.set(key, window.slice(-10));
}


// --- DATABASE INITIALIZATION (SQLITE) ---
async function initDB() {
    db = await open({
        filename: './data/chaka_data.db', // THIS IS YOUR ENTIRE UNLIMITED DATABASE FILE!
        driver: sqlite3.Database
    });

    // --- DURABILITY: WAL journaling survives crashes without corruption, allows
    // concurrent reads during writes, and busy_timeout stops "database is locked"
    // errors under load. synchronous=NORMAL is the recommended pairing with WAL.
    await db.exec(`PRAGMA journal_mode = WAL;`);
    await db.exec(`PRAGMA synchronous = NORMAL;`);
    await db.exec(`PRAGMA busy_timeout = 10000;`);
    await db.exec(`PRAGMA wal_autocheckpoint = 500;`);

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
            google_id TEXT,
            account_type TEXT,
            business_name TEXT,
            objective TEXT,
            tone TEXT,
            onboarded INTEGER DEFAULT 0
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
            active_api TEXT DEFAULT 'deepseek',
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
        CREATE TABLE IF NOT EXISTS knowledge_graph (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            contact_id TEXT,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            timestamp INTEGER,
            embedding TEXT
        );
        CREATE TABLE IF NOT EXISTS scheduled_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            session_id TEXT,
            contact_id TEXT,
            message TEXT,
            send_at_timestamp INTEGER,
            status TEXT DEFAULT 'pending'
        );
        CREATE TABLE IF NOT EXISTS business_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            kind TEXT NOT NULL,          -- 'product' | 'service' | 'info' | 'faq'
            name TEXT NOT NULL,          -- product/service name, info label, or FAQ question
            detail TEXT,                 -- description / answer / value
            price TEXT,                  -- free-text price (optional, products/services)
            created_at INTEGER
        );
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT,
            contact_id TEXT NOT NULL,
            contact_name TEXT,
            status TEXT DEFAULT 'new',   -- new | interested | hot | negotiating | won | lost
            summary TEXT,                -- what they want / intent
            created_at INTEGER,
            updated_at INTEGER,
            UNIQUE(user_id, session_id, contact_id)
        );
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT,
            contact_id TEXT,
            contact_name TEXT,
            type TEXT DEFAULT 'handoff', -- handoff | lead | order | system
            message TEXT,
            handled INTEGER DEFAULT 0,
            created_at INTEGER
        );
        CREATE TABLE IF NOT EXISTS reply_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT,
            contact_id TEXT,
            bot_text TEXT,      -- what the AI last said in that chat
            owner_text TEXT,    -- what the human owner actually wrote right after
            created_at INTEGER
        );
        CREATE TABLE IF NOT EXISTS learned_rules (
            user_id TEXT PRIMARY KEY,
            rules TEXT,             -- distilled standing preferences (plain bullet list)
            feedback_seen INTEGER,  -- highest reply_feedback.id already distilled
            updated_at INTEGER
        );
    `);

    // --- MIGRATION: ADD USER_ID COLUMNS (IGNORES IF EXIST) ---
    try { await db.exec(`ALTER TABLE global_settings ADD COLUMN user_id TEXT DEFAULT 'admin'`); } catch (e) { }
    // Note: SQLite ALTER TABLE ADD COLUMN cannot define composite Primary Keys retroactively, 
    // so we rely on session scoping. It's safe since session IDs are globally unique per user via auth\_baileys_{uid}_{sid}.
    try { await db.exec(`ALTER TABLE global_settings ADD COLUMN active_api TEXT DEFAULT 'deepseek'`); } catch (e) { }
    try { await db.exec(`ALTER TABLE global_settings ADD COLUMN chaka_model TEXT DEFAULT 'chaka-medium'`); } catch (e) { }
    try { await db.exec(`ALTER TABLE contacts ADD COLUMN user_id TEXT DEFAULT 'admin'`); } catch (e) { }
    try { await db.exec(`ALTER TABLE messages ADD COLUMN user_id TEXT DEFAULT 'admin'`); } catch (e) { }
    try { await db.exec(`ALTER TABLE stickers ADD COLUMN user_id TEXT DEFAULT 'admin'`); } catch (e) { }
    try { await db.exec(`ALTER TABLE contacts ADD COLUMN draft_mode INTEGER DEFAULT 0`); } catch (e) { }
    try { await db.exec(`ALTER TABLE global_settings ADD COLUMN global_draft_mode INTEGER DEFAULT 0`); } catch (e) { }
    // --- MIGRATION: ONBOARDING / ACCOUNT-TYPE PROFILE ---
    try { await db.exec(`ALTER TABLE users ADD COLUMN account_type TEXT`); } catch (e) { }
    try { await db.exec(`ALTER TABLE users ADD COLUMN business_name TEXT`); } catch (e) { }
    try { await db.exec(`ALTER TABLE users ADD COLUMN objective TEXT`); } catch (e) { }
    try { await db.exec(`ALTER TABLE users ADD COLUMN tone TEXT`); } catch (e) { }
    try {
        // Runs only on the first boot after this column is added (subsequent ALTERs throw
        // and are caught), so pre-existing accounts are grandfathered in as onboarded and
        // are never forced through the flow or have their custom prompts overwritten.
        await db.exec(`ALTER TABLE users ADD COLUMN onboarded INTEGER DEFAULT 0`);
        await db.exec(`UPDATE users SET onboarded = 1`);
    } catch (e) { }
    // --- MIGRATION: ADMIN ANALYTICS (geolocation + activity) ---
    try { await db.exec(`ALTER TABLE users ADD COLUMN last_ip TEXT`); } catch (e) { }
    try { await db.exec(`ALTER TABLE users ADD COLUMN country TEXT`); } catch (e) { }
    try { await db.exec(`ALTER TABLE users ADD COLUMN country_code TEXT`); } catch (e) { }
    try { await db.exec(`ALTER TABLE users ADD COLUMN city TEXT`); } catch (e) { }
    try { await db.exec(`ALTER TABLE users ADD COLUMN last_active INTEGER`); } catch (e) { }

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
    await db.run(`INSERT OR IGNORE INTO global_settings (id, user_id, master_auto_reply, active_api, chaka_model) VALUES ('settings', ?, 1, 'deepseek', 'chaka-medium')`, [adminId]);
    await loadGlobalConfig(adminId); // Load the admin's config just for legacy compatibility in memory, though we should transition to per-user active_apis.
    console.log("🗄️ Multi-Tenant SQLite Database Initialized Successfully.");
}

async function loadGlobalConfig() {
    try {
        const row = await db.get(`SELECT * FROM global_settings WHERE id = 'settings'`);
        if (row) {
            // API keys are platform-hosted via env (GEMINI_API_KEYS) — never overridden from the DB.
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
async function getRotatedModel(userId, modelName = GEMINI_MODEL) {
    // Keys are hosted centrally by the platform (env GEMINI_API_KEYS) — users never
    // supply their own. userId is kept for signature compatibility / future per-tenant limits.
    if (!API_KEYS || API_KEYS.length === 0) {
        throw new Error("No platform Gemini keys configured (set GEMINI_API_KEYS).");
    }
    const randomKey = API_KEYS[Math.floor(Math.random() * API_KEYS.length)];
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

        // 1. Try to get REAL human-written samples (exclude BOT replies by filtering out sender = MY_NAME)
        let samples = await db.all(
            `SELECT text FROM messages 
             WHERE session_id = ? AND contact_id = ? AND user_id = ? AND is_from_me = 1 
             AND sender != 'BOT' AND message_id NOT LIKE 'BOT_%'
             ORDER BY timestamp DESC LIMIT 8`,
            [sessionId, contactId, userId]
        );

        // 2. Fallback: If no samples for this contact, get general recent style from this session
        if (!samples || samples.length < 3) {
            console.log(`[${sessionId}] 🎭 Limited contact samples, pulling general session style...`);
            samples = await db.all(
                `SELECT text FROM messages 
                 WHERE session_id = ? AND user_id = ? AND is_from_me = 1 
                 AND sender != 'BOT' AND message_id NOT LIKE 'BOT_%'
                 ORDER BY timestamp DESC LIMIT 10`,
                [sessionId, userId]
            );
        }

        if (!samples || samples.length === 0) return "";

        // 3. Clean up samples: remove bot artifacts, duplicates, and very short messages
        const seen = new Set();
        const cleanedSamples = (samples || [])
            .map(s => s.text)
            .filter(t => {
                if (!t || t.length < 3) return false;
                const lower = t.toLowerCase().trim();
                if (seen.has(lower)) return false; // Remove duplicates
                seen.add(lower);
                return !lower.startsWith('[sticker') && !lower.startsWith('bot_');
            })
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
        const model = await getRotatedModel(userId, GEMINI_MODEL);

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

// Transcribe an incoming WhatsApp voice note. Uses Gemini's native audio support
// (same inline-data path as vision) so no local STT model is needed. WhatsApp PTT
// audio is OGG/Opus.
async function transcribeAudio(userId, sessionId, audioBuffer, mimeType = "audio/ogg") {
    try {
        console.log(`[${sessionId}] 🎙️ Transcribing voice note with Gemini...`);
        const model = await getRotatedModel(userId, GEMINI_MODEL);

        const prompt = `Transcribe this WhatsApp voice note to text in its original language. ` +
            `Return ONLY the spoken words — no commentary, labels, or quotes. ` +
            `If there is no intelligible speech, return an empty string.`;

        const result = await model.generateContent([
            prompt,
            {
                inlineData: {
                    data: audioBuffer.toString("base64"),
                    // Strip any codec suffix (e.g. "audio/ogg; codecs=opus") for the API.
                    mimeType: (mimeType || "audio/ogg").split(";")[0].trim(),
                },
            },
        ]);

        const transcript = result.response.text().trim();
        console.log(`[${sessionId}] 🎙️ Transcript: ${transcript.slice(0, 80)}${transcript.length > 80 ? '…' : ''}`);
        return transcript;
    } catch (e) {
        console.error("Audio Transcription Failed:", e.message);
        return "";
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

// --- DEEPSEEK LLM INTEGRATION (PRIMARY — direct paid API) ---
async function tryDeepSeekFailover(userId, sessionId, systemPrompt, userPrompt, opts = {}) {
    if (!DEEPSEEK_API_KEY) {
        console.warn(`[${sessionId}] ⚠️ DEEPSEEK_API_KEY missing — skipping DeepSeek.`);
        return null;
    }

    // Retry transient failures (network blips, timeouts, 429/5xx) so a single hiccup
    // doesn't drop a reply. Auth/validation errors are NOT retried (they won't self-heal).
    const MAX_ATTEMPTS = 3;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        console.log(`[${sessionId}] 🔵 Requesting DeepSeek (${DEEPSEEK_MODEL})${attempt > 1 ? ` [retry ${attempt}/${MAX_ATTEMPTS}]` : ''}...`);
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 30000);
        try {
            const response = await fetch(`${DEEPSEEK_BASE_URL}/chat/completions`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${DEEPSEEK_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: DEEPSEEK_MODEL,
                    messages: [
                        { role: 'system', content: systemPrompt },
                        { role: 'user', content: userPrompt }
                    ],
                    max_tokens: 500,
                    // DeepSeek recommends ~1.3 for natural conversation; callers pass
                    // 1.3 for casual accounts, 1.0 for professional (less improvisation).
                    temperature: typeof opts.temperature === 'number' ? opts.temperature : 1.0,
                    stream: false
                }),
                signal: controller.signal
            });

            // Rate limit / server errors are transient — back off and retry.
            if (response.status === 429 || response.status >= 500) {
                console.warn(`[${sessionId}] ⚠️ DeepSeek transient HTTP ${response.status} (attempt ${attempt}/${MAX_ATTEMPTS}).`);
                if (attempt < MAX_ATTEMPTS) { await delay(800 * attempt); continue; }
                return null;
            }

            const data = await response.json();

            if (data && data.choices && data.choices.length > 0) {
                const content = data.choices[0].message.content;
                if (content && content.trim()) return content.trim();
                console.warn(`[${sessionId}] ⚠️ DeepSeek returned empty content.`);
                return null;
            }
            if (data && data.error) {
                // e.g. invalid key / bad request — retrying won't help.
                console.warn(`[${sessionId}] ⚠️ DeepSeek Error: ${data.error.message || JSON.stringify(data.error)}`);
                return null;
            }
            console.warn(`[${sessionId}] ⚠️ DeepSeek returned no choices (HTTP ${response.status}).`);
            return null;
        } catch (error) {
            // Network failure or timeout/abort — transient, retry.
            console.error(`[${sessionId}] 🔥 DeepSeek Fetch Error (attempt ${attempt}/${MAX_ATTEMPTS}): ${error.message}`);
            if (attempt < MAX_ATTEMPTS) { await delay(800 * attempt); continue; }
            return null;
        } finally {
            clearTimeout(timer);
        }
    }
    return null;
}

// --- OPENROUTER LLM INTEGRATION ---
async function tryOpenRouterFailover(userId, sessionId, systemPrompt, userPrompt) {
    if (!OPENROUTER_API_KEY) {
        console.warn(`[${sessionId}] ⚠️ OpenRouter API Key missing!`);
        return null;
    }

    console.log(`[${sessionId}] 🔄 Requesting OpenRouter (${OPENROUTER_TEXT_MODEL})...`);

    const referer = process.env.PUBLIC_URL || 'http://localhost:3000';

    try {
        const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
                'Content-Type': 'application/json',
                'HTTP-Referer': referer,
                'X-Title': 'WhatsApp Crawler AI'
            },
            body: JSON.stringify({
                model: OPENROUTER_TEXT_MODEL,
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userPrompt }
                ]
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
}

// --- GROQ LLM INTEGRATION (PRIMARY - 14,400 req/day, ~700ms) ---
async function tryGroqFailover(userId, sessionId, systemPrompt, userPrompt) {
    const key = GROQ_API_KEY || process.env.GROQ_API_KEY;
    if (!key) {
        console.warn(`[${sessionId}] ⚠️ Groq API Key missing! Set GROQ_API_KEY in .env`);
        return null;
    }
    console.log(`[${sessionId}] ⚡ Requesting Groq (Llama 3.3 70B)...`);
    try {
        const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${key}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: 'llama-3.3-70b-versatile',
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userPrompt }
                ],
                max_tokens: 200,
                temperature: 0.7
            })
        });
        const data = await response.json();
        if (data.choices?.[0]?.message?.content) {
            const reply = data.choices[0].message.content.trim();
            console.log(`[${sessionId}] ⚡ Groq replied in ~700ms`);
            return reply;
        }
        if (data.error) {
            console.warn(`[${sessionId}] ⚠️ Groq Error: ${data.error.message}`);
            // If rate limited, fall through to next provider
        }
    } catch (e) {
        console.error(`[${sessionId}] 🔥 Groq Fetch Error:`, e.message);
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

async function orchestrateAIResponse(userId, sessionId, systemPrompt, userPrompt, opts = {}) {
    const row = await db.get(`SELECT active_api FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);
    const activeApi = (row && row.active_api) ? row.active_api : 'deepseek';

    // Bounded retry: 2 full rounds through the provider chain, then give up gracefully.
    // (The old while(true) could hold one message hostage forever and starve the queue —
    // WhatsApp redelivers on reconnect anyway, and the caller notifies the owner on failure.)
    const MAX_ROUNDS = 2;
    for (let attempts = 1; attempts <= MAX_ROUNDS; attempts++) {
        console.log(`[${sessionId}] 🛰️ AI Orchestrator: Try #${attempts}/${MAX_ROUNDS}`);

        try {
            if (activeApi === 'deepseek') {
                // DEEPSEEK (direct paid API): primary engine (with its own retries).
                // Fallback chain: DeepSeek → OpenRouter → Gemini (Chaka removed — endpoint is down/400).
                const result = await tryDeepSeekFailover(userId, sessionId, systemPrompt, userPrompt, opts);
                if (result) return result;

                console.warn(`[${sessionId}] DeepSeek failed, falling back to OpenRouter...`);
                const orResult = await tryOpenRouterFailover(userId, sessionId, systemPrompt, userPrompt);
                if (orResult) return orResult;

                console.warn(`[${sessionId}] OpenRouter failed, falling back to Gemini...`);
                const gemResult = await tryGeminiFailover(userId, sessionId, systemPrompt, userPrompt);
                if (gemResult) return gemResult;

            } else if (activeApi === 'qwen') {
                // LLAMA 3.1 (Labelled Qwen/Local)
                const result = await tryQwenFailover(userId, sessionId, systemPrompt, userPrompt);
                if (result) return result;

                // FALLBACK ONLY TO CHAKA IF LLAMA IS DOWN
                console.warn(`[${sessionId}] Llama 3.1 Offline, falling back to Chaka...`);
                const chakaResult = await tryChakaFailover(userId, sessionId, `${systemPrompt}\n\n${userPrompt}`);
                if (chakaResult) return chakaResult;

            } else if (activeApi === 'groq') {
                // GROQ: Primary fast provider, falls back to Gemini
                const result = await tryGroqFailover(userId, sessionId, systemPrompt, userPrompt);
                if (result) return result;

                console.warn(`[${sessionId}] Groq failed, falling back to Gemini...`);
                const gemFallback = await tryGeminiFailover(userId, sessionId, systemPrompt, userPrompt);
                if (gemFallback) return gemFallback;

            } else if (activeApi === 'openrouter') {
                const result = await tryOpenRouterFailover(userId, sessionId, systemPrompt, userPrompt);
                if (result) return result;
                
                console.warn(`[${sessionId}] OpenRouter Offline, falling back to Chaka...`);
                const chakaResult = await tryChakaFailover(userId, sessionId, `${systemPrompt}\n\n${userPrompt}`);
                if (chakaResult) return chakaResult;

            } else if (activeApi === 'gemini') {
                const result = await tryGeminiFailover(userId, sessionId, systemPrompt, userPrompt);
                if (result) return result;
                const chakaResult = await tryChakaFailover(userId, sessionId, `${systemPrompt}\n\n${userPrompt}`);
                if (chakaResult) return chakaResult;
            } else if (activeApi === 'chaka') {
                const result = await tryChakaFailover(userId, sessionId, `${systemPrompt}\n\n${userPrompt}`);
                if (result) return result;

                console.warn(`[${sessionId}] Chaka Offline, falling back to Gemini...`);
                const geminiResult = await tryGeminiFailover(userId, sessionId, systemPrompt, userPrompt);
                if (geminiResult) return geminiResult;
            } else {
                const result = await tryChakaFailover(userId, sessionId, `${systemPrompt}\n\n${userPrompt}`);
                if (result) return result;
                const geminiResult = await tryGeminiFailover(userId, sessionId, systemPrompt, userPrompt);
                if (geminiResult) return geminiResult;
            }

            // Round exhausted — brief pause before the final retry round.
            console.warn(`[${sessionId}] ⚠️ All providers failed on round ${attempts}/${MAX_ROUNDS}.`);
            if (attempts < MAX_ROUNDS) await delay(8000);
        } catch (e) {
            console.error("Orchestrator Loop Error:", e);
            if (attempts < MAX_ROUNDS) await delay(5000);
        }
    }

    console.error(`[${sessionId}] 🚨 AI exhausted all rounds — giving up on this message (owner will be notified).`);
    io.emit('log', { sessionId, msg: "🚨 AI couldn't generate a reply after all retries." });
    return null;
}

// --- SELF-REVIEW: distill owner corrections into standing rules ---
// Periodically (per user, only when there's NEW feedback) the AI studies every case
// where a human owner stepped in or edited a draft, and compresses them into a short
// list of standing preferences. Raw examples fade with recency; distilled rules are
// permanent — this is how the assistant genuinely improves over time.
async function distillLessons() {
    try {
        const users = await db.all(`SELECT user_id, COUNT(*) c, MAX(id) mx FROM reply_feedback GROUP BY user_id HAVING c >= 3`);
        for (const u of users) {
            const prev = await db.get(`SELECT feedback_seen FROM learned_rules WHERE user_id = ?`, [u.user_id]);
            if (prev && prev.feedback_seen >= u.mx) continue; // nothing new to learn

            const rows = await db.all(`SELECT bot_text, owner_text FROM reply_feedback WHERE user_id = ? ORDER BY id DESC LIMIT 20`, [u.user_id]);
            const examples = rows.map(r => `AI wrote: "${(r.bot_text || '').slice(0, 180)}"\nOwner wrote instead: "${(r.owner_text || '').slice(0, 180)}"`).join('\n---\n');
            const sys = `You are a coach improving a WhatsApp AI assistant. Below are cases where the assistant's reply was corrected or replaced by its human owner. Distill the owner's preferences into AT MOST 5 short imperative rules (tone, length, wording, emoji use, language choice, what to offer or avoid). Only include rules clearly supported by the examples. Output ONLY the rules, one per line, each starting with "- ". No preamble.`;

            const out = await tryDeepSeekFailover(u.user_id, 'self-review', sys, examples, { temperature: 1.0 });
            if (out && out.trim().startsWith('-')) {
                await db.run(`INSERT INTO learned_rules (user_id, rules, feedback_seen, updated_at) VALUES (?,?,?,?)
                              ON CONFLICT(user_id) DO UPDATE SET rules=excluded.rules, feedback_seen=excluded.feedback_seen, updated_at=excluded.updated_at`,
                    [u.user_id, out.trim().slice(0, 1200), u.mx, Math.floor(Date.now() / 1000)]);
                console.log(`🧠 [SELF-REVIEW] Distilled ${rows.length} corrections into standing rules for user ${u.user_id.slice(0, 8)}…`);
            }
            await delay(3000); // gentle pacing between users
        }
    } catch (e) { console.error('Self-review failed:', e.message); }
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

async function tryGeminiFailover(userId, sessionId, systemPrompt, userPrompt) {
    console.log(`[${sessionId}] 🔄 Starting Gemini 2.5 Flash...`);

    // Platform-hosted keys only (env GEMINI_API_KEYS). Users do not supply keys.
    const keys = API_KEYS || [];
    if (keys.length === 0) {
        console.warn(`[${sessionId}] ⚠️ No platform Gemini keys configured — skipping Gemini.`);
        return null;
    }

    for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        const genAI = new GoogleGenerativeAI(key);
        
        try {
            console.log(`[${sessionId}] 🛠️ Testing Gemini -> Key #${i + 1} | Model: `);
            const model = genAI.getGenerativeModel({ 
                model: GEMINI_MODEL,
                systemInstruction: systemPrompt,
                generationConfig: { temperature: 0.7, maxOutputTokens: 200 }
            });
            const result = await model.generateContent(userPrompt);
            const text = result.response.text();
            if (text && text.length > 0) return text.trim();
        } catch (e) {
            console.warn(`[${sessionId}] ⚠️ Gemini 2.5 Flash (Key #${i + 1}) Failed: ${e.message.split('\n')[0]}`);
            await delay(2000);
        }
    }
    return null;
}

// --- SERVER SETUP & AUTHENTICATION ---
const app = express();
app.set('trust proxy', 1); // Trust Fly.io proxy for HTTPS redirects

// Allowed origins for CORS / Socket.io. Comma-separated list in CORS_ORIGINS.
// In dev (no list set) we allow all; in prod an explicit allowlist is required.
const CORS_ORIGINS = (process.env.CORS_ORIGINS || '').split(',').map(o => o.trim()).filter(Boolean);
const corsOrigin = CORS_ORIGINS.length > 0 ? CORS_ORIGINS : (IS_PROD ? false : true);
if (IS_PROD && CORS_ORIGINS.length === 0) {
    console.warn("⚠️ CORS_ORIGINS is not set in production — cross-origin requests will be blocked.");
}

// Security headers. CSP is disabled because the dashboard ships inline scripts/styles;
// enable a tailored policy once the front-end is refactored to external assets.
app.use(helmet({ contentSecurityPolicy: false, crossOriginEmbedderPolicy: false }));
app.use(express.json({ limit: '5mb' }));

const server = createServer(app);
const io = new Server(server, {
    cors: {
        origin: corsOrigin,
        methods: ["GET", "POST"]
    }
});
// index:false so "/" falls through to the marketing landing page below instead of
// auto-serving the dashboard (index.html). Explicit paths like /index.html still work.
app.use(express.static('public', { index: false }));

// Throttle authentication endpoints to slow down credential brute-forcing.
const authLimiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 30,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: "Too many attempts. Please try again later." }
});

// Marketing landing page — the public entry point. "Get Started" funnels into signup.
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'landing.html'));
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
            const role = email === ADMIN_EMAIL ? 'admin' : 'user';
            const hash = await bcrypt.hash(uuidv4(), 10);
            await db.run(
                `INSERT INTO users (id, email, password_hash, display_name, role, created_at) VALUES (?, ?, ?, ?, ?, ?)`,
                [userId, email, hash, displayName, role, Math.floor(Date.now() / 1000)]
            );
            await db.run(
                `INSERT OR IGNORE INTO global_settings (id, user_id, master_auto_reply, active_api, chaka_model) VALUES ('settings', ?, 1, 'deepseek', 'chaka-medium')`, 
                [userId]
            );
            user = { id: userId, email, display_name: displayName, role };
        } else if (email === ADMIN_EMAIL && user.role !== 'admin') {
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
            // Go straight into the app (the dashboard's guard routes to onboarding if needed).
            window.location.replace('/index.html');
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
app.post('/api/register', authLimiter, async (req, res) => {
    try {
        const { email, password, displayName } = req.body;
        if (!email || !password) return res.status(400).json({ error: "Email and password required" });
        
        const existing = await db.get('SELECT id FROM users WHERE email = ?', [email]);
        if (existing) return res.status(400).json({ error: "Email already registered" });

        const hash = await bcrypt.hash(password, 10);
        const userId = uuidv4();
        const role = email === ADMIN_EMAIL ? 'admin' : 'user';
        
        await db.run(
            `INSERT INTO users (id, email, password_hash, display_name, role, created_at) VALUES (?, ?, ?, ?, ?, ?)`,
            [userId, email, hash, displayName || email.split('@')[0], role, Math.floor(Date.now() / 1000)]
        );

        // Auto-seed global settings for the new user so the dashboard doesn't crash
        await db.run(
            `INSERT OR IGNORE INTO global_settings (id, user_id, master_auto_reply, active_api, chaka_model) VALUES ('settings', ?, 1, 'deepseek', 'chaka-medium')`, 
            [userId]
        );

        await db.run('UPDATE users SET last_active = ? WHERE id = ?', [Math.floor(Date.now() / 1000), userId]);
        geolocateAndStore(userId, req.ip); // fire-and-forget

        const token = jwt.sign({ id: userId, email, role: role }, JWT_SECRET, { expiresIn: '7d' });
        res.json({ token, user: { id: userId, email, displayName: displayName || email.split('@')[0], role: role } });
    } catch (e) {
        console.error("Register error:", e);
        res.status(500).json({ error: "Internal server error" });
    }
});

app.post('/api/login', authLimiter, async (req, res) => {
    try {
        const { email, password } = req.body;
        const user = await db.get('SELECT * FROM users WHERE email = ?', [email]);
        if (!user || !(await bcrypt.compare(password, user.password_hash))) {
            return res.status(401).json({ error: "Invalid email or password" });
        }
        
        const nowTs = Math.floor(Date.now() / 1000);
        await db.run('UPDATE users SET last_login = ?, last_active = ? WHERE id = ?', [nowTs, nowTs, user.id]);
        geolocateAndStore(user.id, req.ip); // fire-and-forget

        const token = jwt.sign({ id: user.id, email: user.email, role: user.role }, JWT_SECRET, { expiresIn: '7d' });
        res.json({ token, user: { id: user.id, email: user.email, displayName: user.display_name, role: user.role } });
    } catch (e) {
        console.error("Login error:", e);
        res.status(500).json({ error: "Internal server error" });
    }
});

app.get('/api/me', authenticateToken, async (req, res) => {
    const user = await db.get('SELECT id, email, display_name, role, max_sessions, account_type, business_name, objective, tone, onboarded FROM users WHERE id = ?', [req.user.id]);
    res.json(user);
});

// --- ONBOARDING: ACCOUNT TYPE -> TAILORED AI PERSONA ---
const VALID_ACCOUNT_TYPES = ['business', 'personal', 'creator', 'freelancer'];

// Builds the system prompt that shapes how the AI replies on the user's behalf,
// based on the account type and objectives they pick during onboarding.
function buildPersonaPrompt(profile) {
    const {
        account_type = 'personal',
        business_name = '',
        objective = '',
        tone = '',
        display_name = ''
    } = profile || {};

    const name = (business_name || display_name || '').trim();
    const toneLine = tone ? `Preferred tone: ${tone}.` : '';
    const objectiveLine = objective
        ? `What the owner asked you to focus on (work within this, do not add responsibilities beyond it): "${objective}".`
        : `The owner hasn't given detailed instructions yet, so keep it general: greet people warmly, find out what they need, and help with whatever you genuinely know.`;

    // Applies to every business-facing persona. The assistant has NO connected catalog,
    // pricing, or order/booking system unless the owner explicitly described one — so it
    // must never fabricate those. This is what keeps replies honest in production.
    const grounding = (who) => `GROUND RULES — read carefully:
- Only say things you have actually been told (the info above + this conversation). Treat everything else as unknown.
- NEVER invent or assume products, prices, packages, services, stock, availability, opening hours, address, delivery, payment methods, discounts, policies, or an ordering/booking system. None of these exist unless stated above.
- Don't proactively offer orders, bookings, or services the owner hasn't mentioned. Let the other person say what they need first.
- If asked about something you weren't given, do NOT guess. Say you'll check with ${who} and take their question/details, or ask a short clarifying question.
- Never promise or confirm anything on ${who}'s behalf that you can't back up.`;

    const templates = {
        business: `You are the WhatsApp assistant for "${name || 'this business'}", chatting with people who message it, on the owner's behalf.
- Be professional, warm, and genuinely helpful — represent the brand well.
- ${objectiveLine}
- Greet people, understand what they're reaching out about, and help with whatever you actually have information about.
- Keep replies concise and chat-friendly — this is WhatsApp, not email.
${grounding(name || 'the owner')}`,

        freelancer: `You are the WhatsApp assistant for ${name || 'an independent professional'}, handling messages from clients and prospects.
- Be professional, personable, and responsive.
- ${objectiveLine}
- Understand what the person needs and help with what you genuinely know. For specifics you can't confirm (quotes, scope, scheduling), take their details for follow-up.
${grounding(name || 'the owner')}`,

        creator: `You are the WhatsApp assistant for ${name || 'a creator'}, replying to fans and followers.
- Be friendly, appreciative, and engaging — make every fan feel seen.
- ${objectiveLine}
- Thank people for their support and answer what you genuinely know.
- Do NOT invent announcements, drops, release dates, links, prices, or collaborations you weren't told about. If you don't know, say you'll find out.`,

        personal: `You are replying to personal WhatsApp chats on behalf of ${name || 'the user'}, sounding exactly like them.
- Be casual, warm, and natural — like texting a friend. Short messages, emojis where they fit.
- Match the other person's energy and humour. Friendly banter is welcome.
- ${objectiveLine}
- Never sound robotic or corporate. Keep it light and social.
- Don't make up plans, facts, or commitments on their behalf — if you're unsure, keep it vague or say you'll check.`
    };

    const base = templates[account_type] || templates.personal;
    return [base, toneLine].filter(Boolean).join('\n\n');
}

app.post('/api/onboarding', authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;
        let { account_type, business_name, objective, tone } = req.body || {};

        if (!VALID_ACCOUNT_TYPES.includes(account_type)) {
            return res.status(400).json({ error: "Please choose a valid account type." });
        }
        business_name = (business_name || '').toString().trim().slice(0, 120);
        objective = (objective || '').toString().trim().slice(0, 500);
        tone = (tone || '').toString().trim().slice(0, 60);

        const userRow = await db.get('SELECT display_name FROM users WHERE id = ?', [userId]);
        const display_name = userRow?.display_name || '';

        // Persist the profile on the user record.
        await db.run(
            `UPDATE users SET account_type = ?, business_name = ?, objective = ?, tone = ?, onboarded = 1 WHERE id = ?`,
            [account_type, business_name, objective, tone, userId]
        );

        // Generate the tailored AI persona and store it as the user's global custom_prompt,
        // which orchestrateAIResponse() already consumes for every auto-reply.
        const persona = buildPersonaPrompt({ account_type, business_name, objective, tone, display_name });
        const existing = await db.get(`SELECT id FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);
        if (existing) {
            await db.run(`UPDATE global_settings SET custom_prompt = ? WHERE id = 'settings' AND user_id = ?`, [persona, userId]);
        } else {
            await db.run(
                `INSERT INTO global_settings (id, user_id, custom_prompt, master_auto_reply, active_api, chaka_model) VALUES ('settings', ?, ?, 1, 'deepseek', 'chaka-medium')`,
                [userId, persona]
            );
        }

        res.json({ ok: true, account_type, persona });
    } catch (e) {
        console.error("Onboarding error:", e);
        res.status(500).json({ error: "Failed to save onboarding details." });
    }
});

// --- BUSINESS KNOWLEDGE BASE (products, services, info, FAQs) ---
// Real owner-provided facts the AI answers from. The reply path injects these as
// authoritative context so the assistant never has to invent business details.
const VALID_KB_KINDS = ['product', 'service', 'info', 'faq'];

app.get('/api/knowledge', authenticateToken, async (req, res) => {
    const items = await db.all(
        'SELECT id, kind, name, detail, price, created_at FROM business_knowledge WHERE user_id = ? ORDER BY kind, id DESC',
        [req.user.id]
    );
    res.json(items);
});

app.post('/api/knowledge', authenticateToken, async (req, res) => {
    try {
        let { kind, name, detail, price } = req.body || {};
        if (!VALID_KB_KINDS.includes(kind)) return res.status(400).json({ error: 'Invalid item type.' });
        name = (name || '').toString().trim().slice(0, 200);
        detail = (detail || '').toString().trim().slice(0, 2000);
        price = (price || '').toString().trim().slice(0, 80);
        if (!name) return res.status(400).json({ error: 'A name/title is required.' });
        const r = await db.run(
            'INSERT INTO business_knowledge (user_id, kind, name, detail, price, created_at) VALUES (?,?,?,?,?,?)',
            [req.user.id, kind, name, detail, price, Math.floor(Date.now() / 1000)]
        );
        res.json({ ok: true, id: r.lastID });
    } catch (e) {
        console.error('KB add error:', e);
        res.status(500).json({ error: 'Failed to add item.' });
    }
});

app.delete('/api/knowledge/:id', authenticateToken, async (req, res) => {
    await db.run('DELETE FROM business_knowledge WHERE id = ? AND user_id = ?', [req.params.id, req.user.id]);
    res.json({ ok: true });
});

// Bulk import from pasted/CSV text. Each line: "Name <delim> Price <delim> Description"
// (delimiter auto-detected: tab > pipe > comma — so pasting straight from Excel works).
app.post('/api/knowledge/bulk', authenticateToken, async (req, res) => {
    try {
        const text = (req.body?.text || '').toString();
        const kind = VALID_KB_KINDS.includes(req.body?.kind) ? req.body.kind : 'product';
        if (!text.trim()) return res.status(400).json({ error: 'Nothing to import.' });
        const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
        const now = Math.floor(Date.now() / 1000);
        let added = 0;
        for (const line of lines.slice(0, 2000)) {
            const delim = line.includes('\t') ? '\t' : (line.includes('|') ? '|' : ',');
            const parts = line.split(delim).map(p => p.trim());
            const name = parts[0];
            if (!name) continue;
            const price = (parts[1] || '').slice(0, 80);
            const detail = (parts.slice(2).join(', ') || '').slice(0, 2000);
            await db.run('INSERT INTO business_knowledge (user_id, kind, name, detail, price, created_at) VALUES (?,?,?,?,?,?)',
                [req.user.id, kind, name.slice(0, 200), detail, price, now]);
            added++;
        }
        res.json({ ok: true, added });
    } catch (e) {
        console.error('KB bulk error:', e);
        res.status(500).json({ error: 'Bulk import failed.' });
    }
});

// Import from a photo/PDF of a menu or price list — Gemini extracts the products.
app.post('/api/knowledge/import-file', authenticateToken, async (req, res) => {
    try {
        const { dataBase64, mimeType } = req.body || {};
        if (!dataBase64) return res.status(400).json({ error: 'No file provided.' });
        if (!API_KEYS || API_KEYS.length === 0) return res.status(400).json({ error: 'Photo/PDF import needs a Gemini key configured.' });

        const genAI = new GoogleGenerativeAI(API_KEYS[Math.floor(Math.random() * API_KEYS.length)]);
        const model = genAI.getGenerativeModel({ model: GEMINI_MODEL });
        const prompt = `You are extracting a product/menu list from the attached image or document. Return ONLY a JSON array (no markdown, no prose). Each element: {"name": string, "price": string, "detail": string}. Use "" when price or detail is absent. Extract every distinct item you can read. Do NOT invent items that aren't shown.`;
        const result = await model.generateContent([
            prompt,
            { inlineData: { data: dataBase64, mimeType: (mimeType || 'image/jpeg').split(';')[0].trim() } }
        ]);

        let txt = result.response.text().trim().replace(/^```(?:json)?/i, '').replace(/```$/, '').trim();
        let items;
        try { items = JSON.parse(txt); } catch (e) {
            return res.status(422).json({ error: "Couldn't read a product list from that file. Try a clearer photo." });
        }
        if (!Array.isArray(items)) items = [];
        const now = Math.floor(Date.now() / 1000);
        let added = 0;
        for (const it of items.slice(0, 1000)) {
            const name = (it?.name || '').toString().trim();
            if (!name) continue;
            await db.run('INSERT INTO business_knowledge (user_id, kind, name, detail, price, created_at) VALUES (?,?,?,?,?,?)',
                [req.user.id, 'product', name.slice(0, 200), (it.detail || '').toString().slice(0, 2000), (it.price || '').toString().slice(0, 80), now]);
            added++;
        }
        res.json({ ok: true, added });
    } catch (e) {
        console.error('KB file import error:', e);
        res.status(500).json({ error: 'File import failed.' });
    }
});

// Change own password (works for any logged-in user, including admins).
app.post('/api/account/password', authenticateToken, async (req, res) => {
    try {
        const { currentPassword, newPassword } = req.body || {};
        if (!newPassword || newPassword.length < 6) return res.status(400).json({ error: 'New password must be at least 6 characters.' });
        const user = await db.get('SELECT password_hash FROM users WHERE id = ?', [req.user.id]);
        if (!user) return res.status(404).json({ error: 'User not found.' });
        if (user.password_hash) {
            const ok = await bcrypt.compare(currentPassword || '', user.password_hash);
            if (!ok) return res.status(401).json({ error: 'Current password is incorrect.' });
        }
        const hash = await bcrypt.hash(newPassword, 10);
        await db.run('UPDATE users SET password_hash = ? WHERE id = ?', [hash, req.user.id]);
        res.json({ ok: true });
    } catch (e) {
        console.error('Password change error:', e);
        res.status(500).json({ error: 'Failed to change password.' });
    }
});

// --- CONTACTS: AUTO-REPLY (IGNORE LIST) ---
// Search the user's contacts (across all their WhatsApp nodes). q filters by name/number;
// ignored=1 returns only muted contacts. Keeps the UI simple — no need to pick a session.
app.get('/api/contacts', authenticateToken, async (req, res) => {
    const q = (req.query.q || '').toString().trim();
    const ignoredOnly = req.query.ignored === '1';
    const params = [req.user.id];
    let where = 'user_id = ?';
    if (ignoredOnly) where += ' AND auto_reply = 0';
    if (q) { where += ' AND (name LIKE ? OR contact_id LIKE ?)'; params.push(`%${q}%`, `%${q}%`); }
    const rows = await db.all(
        `SELECT session_id, contact_id, name, jid, auto_reply, last_active
         FROM contacts WHERE ${where}
         ORDER BY (auto_reply = 0) DESC, last_active DESC LIMIT 60`, params
    );
    res.json(rows.map(r => ({
        sessionId: r.session_id,
        contactId: r.contact_id,
        name: r.name || r.contact_id,
        number: (r.contact_id || '').split('_')[0],
        ignored: r.auto_reply === 0
    })));
});

// Toggle whether the AI replies to a contact. enabled=false => ignore.
app.post('/api/contacts/auto-reply', authenticateToken, async (req, res) => {
    const { sessionId, contactId, enabled } = req.body || {};
    if (!sessionId || !contactId) return res.status(400).json({ error: 'Missing contact.' });
    const result = await db.run(
        'UPDATE contacts SET auto_reply = ? WHERE user_id = ? AND session_id = ? AND contact_id = ?',
        [enabled ? 1 : 0, req.user.id, sessionId, contactId]
    );
    res.json({ ok: true, changed: result.changes });
});

// Count of currently-ignored contacts (for the badge).
app.get('/api/contacts/ignored-count', authenticateToken, async (req, res) => {
    const row = await db.get('SELECT COUNT(*) as c FROM contacts WHERE user_id = ? AND auto_reply = 0', [req.user.id]);
    res.json({ count: row.c });
});

// --- AGENTS: LEADS & NOTIFICATIONS ---
app.get('/api/leads', authenticateToken, async (req, res) => {
    const leads = await db.all(
        `SELECT id, session_id, contact_id, contact_name, status, summary, created_at, updated_at
         FROM leads WHERE user_id = ? ORDER BY updated_at DESC LIMIT 500`, [req.user.id]
    );
    res.json(leads);
});

app.post('/api/leads/:id/status', authenticateToken, async (req, res) => {
    const status = (req.body?.status || '').toString().toLowerCase();
    if (!['new', 'interested', 'hot', 'negotiating', 'won', 'lost'].includes(status)) return res.status(400).json({ error: 'Invalid status.' });
    await db.run('UPDATE leads SET status = ?, updated_at = ? WHERE id = ? AND user_id = ?', [status, Math.floor(Date.now() / 1000), req.params.id, req.user.id]);
    res.json({ ok: true });
});

app.get('/api/notifications', authenticateToken, async (req, res) => {
    const items = await db.all(
        `SELECT id, session_id, contact_id, contact_name, type, message, handled, created_at
         FROM notifications WHERE user_id = ? ORDER BY handled ASC, created_at DESC LIMIT 200`, [req.user.id]
    );
    res.json(items);
});

app.post('/api/notifications/:id/handled', authenticateToken, async (req, res) => {
    await db.run('UPDATE notifications SET handled = 1 WHERE id = ? AND user_id = ?', [req.params.id, req.user.id]);
    res.json({ ok: true });
});

// --- ADMIN API ENDPOINTS ---
// --- ADMIN HELPERS ---
const isAdminReq = (req) => req.user.role === 'admin' || req.user.email === ADMIN_EMAIL;

// Best-effort IP geolocation (free, no key). For production scale, swap for a paid/offline
// provider (ipinfo/MaxMind). Fire-and-forget so it never slows down auth.
async function geolocateAndStore(userId, ip) {
    try {
        if (!ip || !userId) return;
        const clean = String(ip).replace('::ffff:', '').trim();
        await db.run(`UPDATE users SET last_ip = ? WHERE id = ?`, [clean, userId]);
        // Skip private / loopback addresses — they don't geolocate.
        if (/^(127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|::1|fc00:|fe80:|localhost)/i.test(clean) || clean === '1') return;
        const resp = await fetch(`http://ip-api.com/json/${clean}?fields=status,country,countryCode,city`);
        const data = await resp.json();
        if (data && data.status === 'success') {
            await db.run(`UPDATE users SET country = ?, country_code = ?, city = ? WHERE id = ?`,
                [data.country || null, data.countryCode || null, data.city || null, userId]);
        }
    } catch (e) {
        console.warn('Geolocation failed:', e.message);
    }
}

// Computes which users currently have a live (connected) WhatsApp session.
function getOnlineUserIds() {
    const ids = new Set();
    for (const [globalId, s] of sessions.entries()) {
        if (s && s.isConnected && typeof globalId === 'string') ids.add(globalId.split('_')[0]);
    }
    return ids;
}

app.get('/api/admin/accounts', authenticateToken, async (req, res) => {
    if (!isAdminReq(req)) return res.status(403).json({ error: "Forbidden" });
    const users = await db.all(`
        SELECT u.id, u.email, u.display_name, u.role, u.account_type, u.country, u.country_code, u.city,
               u.created_at, u.last_login, u.last_active,
        (SELECT COUNT(DISTINCT session_id) FROM contacts WHERE user_id = u.id) as session_count
        FROM users u ORDER BY COALESCE(u.last_active, u.last_login, u.created_at) DESC
    `);
    const online = getOnlineUserIds();
    res.json(users.map(u => ({ ...u, online: online.has(u.id) })));
});

app.get('/api/admin/stats', authenticateToken, async (req, res) => {
    if (!isAdminReq(req)) return res.status(403).json({ error: "Forbidden" });
    const now = Math.floor(Date.now() / 1000);
    const dayAgo = now - 86400;

    const totalUsers = (await db.get('SELECT COUNT(*) as c FROM users')).c;
    const totalMessages = (await db.get('SELECT COUNT(*) as c FROM messages')).c;
    const failedRequests = (await db.get(`SELECT COUNT(*) as c FROM request_logs WHERE status = 'failed'`)).c;
    const active24h = (await db.get('SELECT COUNT(*) as c FROM users WHERE COALESCE(last_active, last_login, 0) >= ?', [dayAgo])).c;
    const newToday = (await db.get('SELECT COUNT(*) as c FROM users WHERE created_at >= ?', [dayAgo])).c;

    const online = getOnlineUserIds();
    let liveSessions = 0;
    for (const [, s] of sessions.entries()) if (s && s.isConnected) liveSessions++;

    const byType = await db.all(`SELECT COALESCE(NULLIF(account_type,''),'unset') as type, COUNT(*) as c FROM users GROUP BY type ORDER BY c DESC`);
    const byCountry = await db.all(`SELECT COALESCE(NULLIF(country,''),'Unknown') as country, country_code, COUNT(*) as c FROM users GROUP BY country ORDER BY c DESC LIMIT 50`);

    res.json({
        totalUsers, totalMessages, failedRequests,
        active24h, newToday,
        onlineUsers: online.size, liveSessions,
        byType, byCountry
    });
});

// --- AI OPS ADMIN: ask DeepSeek about the live platform state ---
app.post('/api/admin/ai', authenticateToken, async (req, res) => {
    if (!isAdminReq(req)) return res.status(403).json({ error: "Forbidden" });
    try {
        const question = (req.body?.question || '').toString().slice(0, 1000);
        if (!question) return res.status(400).json({ error: "Ask a question." });

        const now = Math.floor(Date.now() / 1000);
        const dayAgo = now - 86400;
        const totalUsers = (await db.get('SELECT COUNT(*) as c FROM users')).c;
        const active24h = (await db.get('SELECT COUNT(*) as c FROM users WHERE COALESCE(last_active,last_login,0) >= ?', [dayAgo])).c;
        const byType = await db.all(`SELECT COALESCE(NULLIF(account_type,''),'unset') as type, COUNT(*) as c FROM users GROUP BY type`);
        const byCountry = await db.all(`SELECT COALESCE(NULLIF(country,''),'Unknown') as country, COUNT(*) as c FROM users GROUP BY country ORDER BY c DESC LIMIT 15`);
        const recentErrors = await db.all(`SELECT engine, model, status FROM request_logs WHERE status='failed' ORDER BY id DESC LIMIT 15`);
        const online = getOnlineUserIds();
        let liveSessions = 0; for (const [, s] of sessions.entries()) if (s && s.isConnected) liveSessions++;

        const context = {
            totalUsers, activeUsers24h: active24h, onlineUsers: online.size, liveWhatsappSessions: liveSessions,
            accountTypes: byType, topCountries: byCountry, recentFailedAiRequests: recentErrors,
            primaryEngine: 'DeepSeek (deepseek-chat)', visionAudioEngine: GEMINI_MODEL
        };

        const systemPrompt = `You are the AI Operations Admin for "Chaka", a multi-tenant WhatsApp AI assistant SaaS. You help the platform owner oversee and troubleshoot the system. Answer ONLY from the live data provided below — never invent numbers. Be concise, concrete, and proactive: surface anything notable (errors, drop-offs, concentration of users). If the owner asks you to FIX something, explain the exact steps and clearly flag anything that needs a human to execute (you can advise but do not claim to have changed the system). \n\nLIVE PLATFORM DATA:\n${JSON.stringify(context, null, 2)}`;

        const answer = await tryDeepSeekFailover(req.user.id, 'admin-ai', systemPrompt, question);
        res.json({ answer: answer || "I couldn't reach the AI engine right now. Try again in a moment.", context });
    } catch (e) {
        console.error("Admin AI error:", e);
        res.status(500).json({ error: "AI admin failed." });
    }
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

// --- PERSISTENCE: Memory sessions Map held here ---
const sessions = new Map();

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

    // 3b. Handle Voice Notes / Audio (transcribed downstream via Gemini)
    if (msg.audioMessage) {
        return {
            type: 'audio',
            text: null,
            mimeType: msg.audioMessage.mimetype || 'audio/ogg',
            msg: msg.audioMessage
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
    const globalId = `${userId}_${sessionId}`;
    try {
        if (!msg) return null;
        const content = extractContent(msg.message);
        const jid = msg.key.remoteJid;

        if (!content.text && content.type !== 'sticker' && content.type !== 'image' && content.type !== 'audio') {
            if (!forceSkipEmbedding) console.log(`[${globalId}] ⏭ Skipping: No text content found.`);
            return null;
        }

        if (jid === 'status@broadcast' || jid.includes('@g.us')) {
            if (!forceSkipEmbedding) console.log(`[${globalId}] ⏭ Skipping: Non-personal chat (${jid})`);
            return null;
        }

        const session = sessions.get(globalId);
        const cleanId = jid.replace(/[^a-zA-Z0-9]/g, "_");

        // --- NEW: VISION ENGINE ---
        if (content.type === 'image') {
            try {
                console.log(`[${globalId}] 👁️ Image detected. Fetching context for occasion analysis...`);

                // Fetch last 5 messages for context
                const recentContextRows = await db.all(
                    `SELECT sender, text FROM messages WHERE session_id = ? AND contact_id = ? AND user_id = ? ORDER BY timestamp DESC LIMIT 5`,
                    [sessionId, cleanId, userId]
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

        // --- VOICE NOTE ENGINE ---
        if (content.type === 'audio') {
            try {
                console.log(`[${globalId}] 🎙️ Voice note detected. Transcribing...`);
                const buffer = await downloadMediaMessage(msg, 'buffer', {});
                if (buffer) {
                    const transcript = await transcribeAudio(userId, sessionId, buffer, content.mimeType);
                    content.text = transcript
                        ? `[VOICE NOTE: ${transcript}]`
                        : `[VOICE NOTE]`;
                } else {
                    content.text = `[VOICE NOTE]`;
                }
            } catch (e) {
                console.error("Voice transcription failed:", e.message);
                content.text = `[VOICE NOTE]`;
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
            msgId, sessionId, cleanId, userId, dbText,
            msg.key.fromMe ? (isBotSentText(sessionId, cleanId, content.text) ? 'BOT' : MY_NAME) : contactName,
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

async function scrapeChatHistory(userId, sessionId, jid) {
    const globalId = `${userId}_${sessionId}`;
    const session = sessions.get(globalId);
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
            const res = await saveMessageToDB(userId, sessionId, msg, true);
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

// --- KNOWLEDGE GRAPH EXTRACTOR ---
async function extractKnowledgeTriples(userId, sessionId, contactId) {
    try {
        const globalId = `${userId}_${sessionId}`;
        // Fetch the last 8 messages
        const messages = await db.all(
            `SELECT sender, text FROM messages 
             WHERE session_id = ? AND contact_id = ? AND user_id = ? 
             ORDER BY timestamp DESC LIMIT 8`, 
            [sessionId, contactId, userId]
        );
        if (messages.length < 2) return;
        messages.reverse();
        
        const conversationScript = messages.map(m => `${m.sender}: ${m.text}`).join('\n');
        
        const systemPrompt = `You are a background memory graph extractor. Extract permanent facts, user preferences, AND the user's specific texting style for this contact from the conversation.
Return ONLY a valid JSON array of arrays, representing [Subject, Predicate, Object] triples.
Example output:
[["User", "loves", "coffee"], ["User Style", "uses emojis", "💀 and 😭"], ["User Tone", "is", "highly informal and uses lowercase"], ["John", "is", "User's boss"]]
If no new permanent facts or style rules are found, return []. Do not include transient states.`;

        const reply = await orchestrateAIResponse(userId, globalId, systemPrompt, conversationScript);
        if (!reply) return;
        
        let jsonStr = reply;
        const match = reply.match(/\[.*\]/s);
        if (match) jsonStr = match[0];
        
        const triples = JSON.parse(jsonStr);
        if (Array.isArray(triples) && triples.length > 0) {
            console.log(`[${sessionId}] 🧠 Graph Extracted: ${triples.length} new facts`);
            for (const triple of triples) {
                if (Array.isArray(triple) && triple.length >= 3) {
                    await db.run(
                        `INSERT INTO knowledge_graph (session_id, contact_id, subject, predicate, object, timestamp) VALUES (?, ?, ?, ?, ?, ?)`,
                        [sessionId, contactId, String(triple[0]), String(triple[1]), String(triple[2]), Math.floor(Date.now() / 1000)]
                    );
                }
            }
        }
    } catch (e) {
        // Silently fail if JSON parse fails or extraction fails
    }
}

// --- HERMES / LOCAL AGENT PIPELINE (Phase 2) ---
async function searchWeb(query) {
    try {
        console.log(`[AGENT] 🌐 Performing background web search for: "${query}"`);
        const response = await fetch('https://html.duckduckgo.com/html/?q=' + encodeURIComponent(query), {
            headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' }
        });
        const html = await response.text();
        const $ = cheerio.load(html);
        let results = [];
        $('.result__snippet').each((i, el) => {
            if (i < 3) results.push($(el).text().trim());
        });
        if (results.length > 0) return results.join(' | ');
        return "No clear results found.";
    } catch (e) {
        console.error("[AGENT] Web search failed:", e.message);
        return "Search failed.";
    }
}

async function runLocalAgentRouter(incomingText) {
    // We use the local ollama instance (e.g. qwen2.5-coder:7b) for zero-cost routing
    try {
        const prompt = `You are an internal routing agent. Analyze this incoming message: "${incomingText}". 
Does the user explicitly ask to look up facts, search the web, check news, or require real-time information? If the answer requires any external information you do not have, you MUST return true.
Respond ONLY with a JSON object. No markdown, no conversational text.
Format: {"requires_search": true, "search_query": "the concise search query to look up"}`;
        
        const response = await fetch('http://localhost:11434/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'qwen2.5-coder:7b',
                prompt: prompt,
                stream: false
            })
        });
        const data = await response.json();
        const rawJson = data.response.trim().replace(/```json/g, '').replace(/```/g, '').trim();
        const parsed = JSON.parse(rawJson);
        return parsed;
    } catch (e) {
        // Router failed, safe default
        return { requires_search: false };
    }
}

async function generateSmartReply(userId, sessionId, contactName, contactId, incomingMsg) {
    const globalId = `${userId}_${sessionId}`;

    // ══════════════════════════════════════════════════
    // SMART TOKEN BUDGET SYSTEM
    // Groq: 70,000 TPM | Gemini: 250,000 TPM
    // We track per-user per-minute spend and scale
    // context/memory dynamically to never hit limits
    // ══════════════════════════════════════════════════
    const currentSpend = getCurrentMinuteSpend(globalId);

    // Detect which engine is active to set the right ceiling
    const engineRow = await db.get(`SELECT active_api FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);
    const activeEngine = engineRow?.active_api || 'groq';
    const MAX_TPM = activeEngine === 'gemini' ? 200000 : 55000; // Conservative safe ceilings

    if (currentSpend > MAX_TPM) {
        console.log(`[${sessionId}] 🚫 TOKEN BUDGET EXHAUSTED (${currentSpend}/${MAX_TPM}) — skipping reply`);
        return null;
    }

    const availableTokens = MAX_TPM - currentSpend;

    // Dynamic scaling: the more budget left, the richer the context
    let contextLimit, memoryLimit, includeStyle;
    if (availableTokens > MAX_TPM * 0.7) {
        contextLimit = 12; memoryLimit = 3; includeStyle = true;
    } else if (availableTokens > MAX_TPM * 0.4) {
        contextLimit = 7;  memoryLimit = 2; includeStyle = true;
    } else if (availableTokens > MAX_TPM * 0.2) {
        contextLimit = 4;  memoryLimit = 1; includeStyle = false;
    } else {
        contextLimit = 3;  memoryLimit = 0; includeStyle = false;
    }

    console.log(`[${sessionId}] 💰 Budget: ${availableTokens}/${MAX_TPM} | CTX:${contextLimit} MEM:${memoryLimit} STYLE:${includeStyle}`);

    let agentContext = "";
    try {
        const agentDecision = await runLocalAgentRouter(incomingMsg);
        if (agentDecision && agentDecision.requires_search && agentDecision.search_query) {
            io.emit('log', { sessionId, msg: `🤖 Agent performing web search for: ${agentDecision.search_query}` });
            const searchResults = await searchWeb(agentDecision.search_query);
            agentContext = `[WEB SEARCH RESULTS for "${agentDecision.search_query}"]\n${searchResults}\n\n`;
            io.emit('log', { sessionId, msg: `✅ Search complete. Injected into AI context.` });
        }
    } catch (e) { console.error("Agent error:", e); }

    try {
        const contactRow = await db.get(`SELECT custom_prompt FROM contacts WHERE session_id = ? AND contact_id = ? AND user_id = ?`, [sessionId, contactId, userId]);
        const globalRow = await db.get(`SELECT * FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);

        // Account type drives tone: business/freelancer stay professional and on-persona;
        // personal/creator mirror the owner's casual texting style.
        const userRow = await db.get(`SELECT account_type FROM users WHERE id = ?`, [userId]);
        const accountType = userRow?.account_type || 'personal';
        const isProfessional = accountType === 'business' || accountType === 'freelancer';

        console.log(`[${sessionId}] 🧠 AI Brain: Engine=${activeEngine.toUpperCase()} | Type=${accountType} | Contact Settings: ${contactRow ? 'Found' : 'Default'}`);

        // ── RECENT CONVERSATION (scoped to THIS contact only) ──
        // CRITICAL: Exclude bot-generated messages to prevent the AI from mimicking its own prior outputs
        const recentHistoryRows = await db.all(
            `SELECT sender, text, message_id FROM messages 
             WHERE session_id = ? AND contact_id = ? AND user_id = ? 
             AND message_id NOT LIKE 'BOT_%'
             ORDER BY timestamp DESC LIMIT ?`,
            [sessionId, contactId, userId, contextLimit]
        );
        const recentHistory = recentHistoryRows.reverse();

        // ══════════════════════════════════════════════════
        // ADVANCED 3-LAYER MEMORY SYSTEM
        //
        // Layer 1 — Keyword Memory (free, instant, no AI)
        //   Fast keyword match for names, dates, facts
        //
        // Layer 2 — Semantic RAG (local AI embeddings)
        //   Finds contextually similar past messages
        //   Only runs if local embedder is available
        //   Scoped to THIS contact's messages only
        //
        // Layer 3 — Long-term Summary (pinned facts)
        //   Key facts extracted and stored separately
        //   Never expires, zero tokens to store
        // ══════════════════════════════════════════════════

        // LAYER 1: Keyword Memory — free, instant
        const keywords = incomingMsg.toLowerCase().split(/\s+/).filter(w => w.length > 3);
        let keywordMemories = [];
        if (keywords.length > 0) {
            const placeholders = keywords.map(() => `text LIKE ?`).join(' OR ');
            const params = keywords.map(k => `%${k}%`);
            keywordMemories = await db.all(
                `SELECT sender, text, date FROM messages 
                 WHERE session_id = ? AND contact_id = ? AND user_id = ? 
                 AND (${placeholders})
                 AND text != ?
                 ORDER BY timestamp DESC LIMIT 3`,
                [sessionId, contactId, userId, ...params, incomingMsg]
            );
        }

        // LAYER 2: Semantic RAG — scoped to contact, top 50 only
        let retrievedMemories = [];
        if (memoryLimit > 0 && localEmbedder) {
            const expandedQuery = expandSlang(incomingMsg);
            const queryVector = await generateLocalEmbedding(expandedQuery);

            if (queryVector) {
                // KEY FIX: Scoped to THIS contact, not all users, limited to 50
                const contactVectors = await db.all(
                    `SELECT message_id, text, timestamp, date, sender, embedding 
                     FROM messages 
                     WHERE session_id = ? AND contact_id = ? AND user_id = ? 
                     AND embedding IS NOT NULL 
                     ORDER BY timestamp DESC LIMIT 50`,
                    [sessionId, contactId, userId]
                );

                if (contactVectors.length > 0) {
                    const parsedQuery = parseEmbedding(queryVector);
                    retrievedMemories = contactVectors
                        .map(msg => ({
                            ...msg,
                            score: cosineSimilarity(parsedQuery, parseEmbedding(msg.embedding))
                        }))
                        .filter(m => m.score > 0.5 && m.text !== incomingMsg)
                        .sort((a, b) => b.score - a.score)
                        .slice(0, memoryLimit);
                }
            }
        }

        // Build memory block — compact format to save tokens
        let memoryDetails = "";
        
        if (agentContext) {
            memoryDetails += agentContext;
        }

        // LAYER 3: GraphRAG Knowledge Graph Fetch
        let graphFacts = [];
        try {
            if (keywords.length > 0) {
                const placeholders = keywords.map(() => `(subject LIKE ? OR predicate LIKE ? OR object LIKE ?)`).join(' OR ');
                const params = keywords.flatMap(k => [`%${k}%`, `%${k}%`, `%${k}%`]);
                graphFacts = await db.all(
                    `SELECT subject, predicate, object FROM knowledge_graph 
                     WHERE session_id = ? AND contact_id = ? 
                     AND (${placeholders})
                     ORDER BY timestamp DESC LIMIT 5`, 
                    [sessionId, contactId, ...params]
                );
            }
            // Always include a few general facts if few keyword matches
            if (graphFacts.length < 5) {
                const extraFacts = await db.all(
                    `SELECT subject, predicate, object FROM knowledge_graph 
                     WHERE session_id = ? AND contact_id = ? 
                     ORDER BY timestamp DESC LIMIT ?`, 
                    [sessionId, contactId, 5 - graphFacts.length]
                );
                graphFacts = graphFacts.concat(extraFacts);
            }
            if (graphFacts.length > 0) {
                const uniqueFacts = Array.from(new Set(graphFacts.map(f => `${f.subject} -> ${f.predicate} -> ${f.object}`)));
                memoryDetails += `[KNOWN FACTS]\n${uniqueFacts.join('\n')}\n\n`;
            }
        } catch(e) { console.error("Graph fetch err:", e); }

        if (keywordMemories.length > 0) {
            const kwBlock = keywordMemories.map(m => `${m.sender}: ${m.text}`).join('\n');
            memoryDetails += `[KEYWORD MATCH]\n${kwBlock}\n\n`;
        }

        for (const mem of retrievedMemories) {
            // Only fetch tight context window (±60s) to keep tokens low
            const context = await db.all(
                `SELECT sender, text FROM messages 
                 WHERE session_id = ? AND contact_id = ? AND user_id = ?
                 AND timestamp BETWEEN ? AND ?
                 ORDER BY timestamp ASC LIMIT 4`,
                [sessionId, contactId, userId, mem.timestamp - 60, mem.timestamp + 60]
            );
            if (context.length > 0) {
                memoryDetails += `[RELATED (${Math.round(mem.score * 100)}% match)]\n`;
                memoryDetails += context.map(c => `${c.sender}: ${c.text}`).join('\n') + '\n\n';
            }
        }

        // LAYER 3: Long-term pinned facts (name, preferences, key events)
        const pinnedFacts = await db.all(
            `SELECT text FROM messages 
             WHERE session_id = ? AND contact_id = ? AND user_id = ? AND is_from_me = 0
             AND (text LIKE '%my name%' OR text LIKE '%I am%' OR text LIKE '%I''m%' 
                  OR text LIKE '%I work%' OR text LIKE '%I live%' OR text LIKE '%I have%'
                  OR text LIKE '%my number%' OR text LIKE '%remember%' OR text LIKE '%birthday%')
             ORDER BY timestamp ASC LIMIT 5`,
            [sessionId, contactId, userId]
        );
        let pinnedContext = "";
        if (pinnedFacts.length > 0) {
            pinnedContext = `[KNOWN FACTS ABOUT ${contactName.toUpperCase()}]\n` +
                pinnedFacts.map(f => `• ${f.text}`).join('\n') + '\n\n';
        }

        // ── STYLE LEARNING (only when budget allows) ──
        // NOTE: Style context is kept minimal to prevent feedback loops
        // Professional accounts follow the persona's tone, NOT the owner's casual style,
        // so we don't feed casual style samples that would pull replies off-brand.
        let styleContext = "";
        if (includeStyle && !isProfessional) {
            const styleSamples = await fetchUserStyleSamples(userId, sessionId, contactId);
            if (styleSamples) {
                styleContext = `[STYLE REFERENCE — imitate the VIBE only: how you capitalise, slang, emoji habits, message length. These are NOT scripts. Do NOT reuse their words or send any of these lines back. Write a fresh reply.]\n${styleSamples}\n\n`;
            }
        }

        // Build conversation script with dedup (strip consecutive identical messages)
        const dedupedHistory = [];
        let lastText = '';
        for (const h of recentHistory) {
            const currentText = (h.text || '').trim().toLowerCase();
            if (currentText && currentText !== lastText) {
                dedupedHistory.push(h);
                lastText = currentText;
            }
        }
        const conversationScript = dedupedHistory.map(h => `${h.sender}: ${h.text}`).join('\n');
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

        // --- USER-DEFINED PERSONA (authoritative) ---
        const userPersona = contactRow?.custom_prompt || globalRow?.custom_prompt || "You are a helpful assistant.";

        // --- BUSINESS KNOWLEDGE BASE injection (real owner-provided facts) ---
        let kbContext = '';
        try {
            const kbItems = await db.all('SELECT kind, name, detail, price FROM business_knowledge WHERE user_id = ? LIMIT 200', [userId]);
            if (kbItems.length) {
                const offerings = kbItems.filter(i => i.kind === 'product' || i.kind === 'service')
                    .map(i => `- ${i.name}${i.price ? ` — ${i.price}` : ''}${i.detail ? `: ${i.detail}` : ''}`);
                const info = kbItems.filter(i => i.kind === 'info').map(i => `- ${i.name}: ${i.detail || ''}`);
                const faqs = kbItems.filter(i => i.kind === 'faq').map(i => `- Q: ${i.name}\n  A: ${i.detail || ''}`);
                kbContext = `\n\n[BUSINESS INFO — these are the ONLY verified facts you know about this business. Answer questions using these. If something a customer asks about is NOT listed here, do not invent it — say you'll check with the owner.]`;
                if (offerings.length) kbContext += `\nProducts / Services:\n${offerings.join('\n')}`;
                if (info.length) kbContext += `\nBusiness details:\n${info.join('\n')}`;
                if (faqs.length) kbContext += `\nFAQs:\n${faqs.join('\n')}`;
            }
        } catch (e) { /* KB fetch failed — fall back to grounding rules */ }

        // --- LEARNING: owner corrections (highest-value signal we have) ---
        // When the human owner stepped into a chat after the AI replied, their message
        // is ground truth. Show the model the most recent examples so it converges on
        // how the owner actually talks and handles situations — this contact first,
        // then account-wide.
        let learnContext = '';
        try {
            // Distilled standing rules (compounded from ALL past corrections) — permanent.
            const ruleRow = await db.get(`SELECT rules FROM learned_rules WHERE user_id = ?`, [userId]);
            if (ruleRow?.rules) {
                learnContext += `\n\n[YOUR OWNER'S STANDING PREFERENCES — learned from every time they corrected you. Follow these strictly:\n${ruleRow.rules}]`;
            }
            // Plus the freshest raw examples (recency signal), this contact first.
            const lessons = await db.all(
                `SELECT bot_text, owner_text FROM reply_feedback WHERE user_id = ?
                 ORDER BY (contact_id = ?) DESC, created_at DESC LIMIT 3`,
                [userId, contactId]);
            if (lessons.length) {
                const lines = lessons.map(l =>
                    `- You said: "${(l.bot_text || '').slice(0, 140)}" → your owner then stepped in and wrote: "${(l.owner_text || '').slice(0, 140)}"`).join('\n');
                learnContext += `\n\n[LEARNED FROM YOUR OWNER — when they stepped into chats, this is how THEY handled it. Match their judgment, tone and wording style (do not copy verbatim):\n${lines}]`;
            }
        } catch (e) { /* learning injection is best-effort */ }

        const scheduleRule = `- If the user asks you to remind them or someone else to do something later, output exactly: [SCHEDULE_MESSAGE: <YYYY-MM-DDTHH:MM:SSZ> | <ContactName OR "CURRENT"> | <The reminder message to send later>]. Use "CURRENT" if it's for the person you are currently talking to. Also include a brief immediate text acknowledging the schedule.`;

        // Agent actions: hidden control tags the assistant appends AFTER its normal reply.
        // The system strips them before sending, so the customer never sees them.
        const agentActions = `- AGENT ACTIONS (silent — the customer must NEVER see these; put them on their own line at the very END of your message; still write a normal helpful reply above them):
  • If the person shows buying interest, asks about a product/price, wants to order/book, or shares order/contact details, append: [LEAD: status=<interested|hot|negotiating|won|lost> | summary=<max 12 words on what they want>]
  • If the chat needs the human owner — an upset/angry customer, a complaint or refund, a complex or high-value request, or they explicitly ask for a human — append: [HANDOFF: <short reason>]
  Only use a tag when it genuinely applies. Never mention these tags in your words.`;

        // Persona is the source of truth. For professional accounts we explicitly forbid
        // the model from copying the other person's casual tone (the cause of off-brand
        // replies like "yo what's good" from a business assistant).
        const personaHeader = isProfessional
            ? `You MUST stay fully in character as defined below at ALL times. This role and its tone are authoritative and override the casual style of anyone you're talking to.\n\n${userPersona}`
            : userPersona;

        const rulesBlock = isProfessional
            ? `[RULES]\n- Always keep the role, voice, and tone described above. Never slip into slang, text-speak, or all-lowercase "lazy" texting, even if the other person does.\n- Be professional, warm, and genuinely helpful. Use correct grammar, spelling, and capitalization.\n- Keep replies concise (1-3 sentences) but complete — actually answer what was asked.\n- NEVER invent or assume business details — products, prices, services, stock, availability, hours, address, delivery, ordering or booking systems, discounts, or policies. Only state what's in your instructions above. If you weren't told it, you don't offer it.\n- Do NOT proactively pitch orders, bookings, or services. Let the person say what they need; if it's something you have no info on, offer to take their details for the owner to follow up.\n- If you don't know a specific detail, say you'll check rather than guessing or making it up.\n- Reply with ONE message only. Never repeat yourself or echo earlier phrases verbatim.\n- Light, fitting emoji use is okay; don't overdo it.\n${scheduleRule}\n${agentActions}`
            : `[RULES]\n- FIRST, actually read their last message and respond to THAT specifically. Your reply must make sense as a direct answer to what they just said — not a generic greeting.\n- Match the WRITING STYLE of the style reference (casual lowercase, slang, emoji habits) — but NEVER copy its words or send back any line from it. Generate fresh wording every time.\n- VARY your replies. Never send the same thing twice or recycle a phrase you already used in this chat (e.g. don't keep saying "just chillin wbu"). If you already said something, say something different.\n- Keep it short and human (1-2 lines), like a real person texting a friend. No AI/assistant vibes.\n- Don't address them by a name unless they told you their name in this conversation.\n- Move the conversation forward: react to what they said, then optionally add a thought — don't just bounce the same question back.\n- Light emoji use only when it fits. If you genuinely have nothing to add, a short natural reply is fine.\n- If you don't know something, say so briefly. Don't make things up.\n${scheduleRule}`;

        let systemInstruction = `[CURRENT TIME: ${new Date().toISOString()}]\n[IDENTITY] ${CORE_IDENTITY}\n\n${personaHeader}${kbContext}${learnContext}\n\n[CONTEXT]\nContact: ${contactName}${stickerContext}\n\n${rulesBlock}`;

        // ── BUILD FINAL PROMPT (token-efficient format) ──
        let userPrompt = '';
        if (pinnedContext) userPrompt += pinnedContext;
        if (memoryDetails) userPrompt += memoryDetails;
        if (styleContext)  userPrompt += styleContext;
        userPrompt += `[RECENT CONVERSATION]\n${conversationScript}\n${MY_NAME}:`;

        // ✂️ SMART TRUNCATION — per-engine token limits
        // Groq: ~6000 chars safe | Gemini: ~12000 | Qwen: ~3500
        const engineLimits = { groq: 6000, gemini: 12000, qwen: 3500, openrouter: 6000, chaka: 4000 };
        const maxPromptLen = engineLimits[activeEngine] || 6000;
        if (userPrompt.length > maxPromptLen) {
            // Preserve recent conversation, trim from top
            const recentConvIdx = userPrompt.lastIndexOf('[RECENT CONVERSATION]');
            if (recentConvIdx > 0 && userPrompt.length - recentConvIdx < maxPromptLen) {
                // Keep all recent conversation, trim only memory/style from top
                const toTrim = userPrompt.length - maxPromptLen;
                userPrompt = userPrompt.substring(toTrim);
            } else {
                userPrompt = userPrompt.substring(userPrompt.length - maxPromptLen);
            }
            console.log(`[${sessionId}] ✂️ Prompt trimmed to ${userPrompt.length} chars for ${activeEngine}`);
        }

        console.log(`[${sessionId}] 📤 GENERATING RESPONSE (System: ${systemInstruction.length}, User: ${userPrompt.length})`);

        const startTime = Date.now();
        // Casual accounts get a warmer temperature (DeepSeek's own guidance for
        // conversation is ~1.3); professional stays precise at 1.0.
        let replyText = await orchestrateAIResponse(userId, globalId, systemInstruction, userPrompt, { temperature: isProfessional ? 1.0 : 1.3 });

        if (!replyText) {
            console.error(`[${globalId}] 🔥 Orchestrator returned null after all retries.`);
            // Surface the failure to the owner instead of failing silently — this chat
            // needs a human until the AI is reachable again.
            try {
                await db.run(`INSERT INTO notifications (user_id, session_id, contact_id, contact_name, type, message, created_at) VALUES (?,?,?,?,?,?,?)`,
                    [userId, sessionId, contactId, contactName, 'system', `AI couldn't reply to ${contactName} (provider unreachable). They may be waiting on you.`, Math.floor(Date.now() / 1000)]);
                io.emit('notification', { sessionId, contactId, contactName, type: 'system', message: `AI couldn't reply to ${contactName} — tap in.`, timestamp: Date.now() });
            } catch (e) { /* notification best-effort */ }
            return null;
        }

        // Track spending and burst stats
        const usedTokens = Math.ceil(userPrompt.length / 4) + Math.ceil(replyText.length / 4);
        trackTokenSpend(globalId, usedTokens);
        recordMessage(userId, contactId, usedTokens);

        console.log(`[${sessionId}] ✅ Generated in ${Date.now() - startTime}ms | Used: ${usedTokens} tokens`);
        console.log(`[${sessionId}] ✨ AI Response: ${replyText}`);
        io.emit('log', { sessionId, msg: `✨ AI Response Generated (${replyText.length} chars)` });

        // Cleanup any accidental formatting
        if (replyText.startsWith(`${MY_NAME}:`)) replyText = replyText.replace(`${MY_NAME}:`, '').trim();
        if (replyText.startsWith(`"`)) replyText = replyText.replace(/^"|"$/g, '').trim();

        // REPETITION GUARD: Detect and strip repeated lines
        const lines = replyText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
        const seen = new Set();
        const uniqueLines = [];
        for (const line of lines) {
            const normalized = line.toLowerCase();
            if (!seen.has(normalized)) {
                seen.add(normalized);
                uniqueLines.push(line);
            }
        }
        replyText = uniqueLines.join('\n').trim();

        // If after dedup the reply is empty or too short, return null
        if (!replyText || replyText.length < 2) return null;

        // RESPONSE VALIDATION: Reject if this response matches any previous bot reply to this contact
        try {
            const recentBotReplies = await db.all(
                `SELECT text FROM messages 
                 WHERE session_id = ? AND contact_id = ? AND user_id = ? 
                 AND message_id LIKE 'BOT_%'
                 ORDER BY timestamp DESC LIMIT 10`,
                [sessionId, contactId, userId]
            );
            const replyLower = replyText.toLowerCase().trim();
            const isDuplicateReply = recentBotReplies.some(r => 
                r.text && r.text.toLowerCase().trim() === replyLower
            );
            if (isDuplicateReply) {
                console.warn(`[${sessionId}] 🚨 BLOCKED: AI tried to repeat a previous response. Retrying with anti-repeat instruction...`);
                // Retry ONCE with explicit anti-repetition prompt
                const retryPrompt = userPrompt + `\n\n[CRITICAL: Your previous response "${replyText}" was already sent before. You MUST say something COMPLETELY DIFFERENT. Respond to the latest message naturally.]`;
                let retryText = await orchestrateAIResponse(userId, globalId, systemInstruction, retryPrompt);
                if (retryText) {
                    if (retryText.startsWith(`${MY_NAME}:`)) retryText = retryText.replace(`${MY_NAME}:`, '').trim();
                    if (retryText.startsWith(`"`)) retryText = retryText.replace(/^"|"$/g, '').trim();
                    const retryLower = retryText.toLowerCase().trim();
                    if (!recentBotReplies.some(r => r.text && r.text.toLowerCase().trim() === retryLower)) {
                        console.log(`[${sessionId}] ✅ Retry produced a fresh response.`);
                        return retryText;
                    }
                }
                console.warn(`[${sessionId}] 🚨 Retry also repeated. Suppressing to avoid spam.`);
                return null;
            }
        } catch (e) {
            console.error(`[${sessionId}] Response validation error:`, e.message);
        }

        return replyText;
    } catch (error) {
        console.error(`[${sessionId}] AI Error:`, error);
        return null;
    }
}

async function startSession(userId, sessionId) {
    const globalId = `${userId}_${sessionId}`;
    
    // Safety guard for malformed IDs
    if (!sessionId || sessionId === 'undefined' || !userId) {
        console.error(`[SYSTEM] Aborting startSession for invalid ID: user=${userId}, session=${sessionId}`);
        return;
    }

    const session = sessions.get(globalId);
    if (session && (session.connectionState === 'connecting' || session.connectionState === 'qr_ready' || session.isConnected)) {
        console.log(`[${globalId}] ⏳ Already initializing/active. Skipping startSession request.`);
        return;
    }

    // Set immediate placeholder to lock the session
    sessions.set(globalId, { isConnected: false, connectionState: 'connecting', handshakeTimer: null });

    console.log(`🚀 Starting Global Session: ${globalId}...`);
    const authPath = `./data/auth_baileys_${globalId}`;
    const { state, saveCreds } = await useMultiFileAuthState(authPath);

    // Fetch the CURRENT WhatsApp Web version. A stale/hard-coded version makes the
    // server stall the Noise handshake (no QR ever appears), so we resolve it live and
    // only fall back to a known-recent build if the lookup fails.
    let version, isLatest;
    try {
        ({ version, isLatest } = await fetchLatestBaileysVersion());
    } catch (e) {
        version = [2, 3000, 1035194821];
        isLatest = false;
        console.warn(`[${globalId}] ⚠️ Could not fetch latest WA version, using fallback ${version.join('.')}: ${e.message}`);
    }
    console.log(`[${globalId}] 📦 WA Version: ${version.join('.')} (Latest: ${isLatest})`);

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
        syncFullHistory: false, 
        browser: Browsers.macOS('Desktop'),
        generateHighQualityLinkPreview: false,
        markOnlineOnConnect: false,
        connectTimeoutMs: 180000,
        keepAliveIntervalMs: 20000, 
        getMessage: async (key) => (await store.loadMessage(key.remoteJid, key.id))?.message || undefined
    });

    store.bind(sock.ev);
    sessions.get(globalId).sock = sock;
    sessions.get(globalId).store = store;
    sessions.get(globalId).phonebook = phonebook;

    // --- CONNECTION WATCHDOG --- (Ensures we don't hang in 'connecting' forever)
    if (sessions.get(globalId).handshakeTimer) clearTimeout(sessions.get(globalId).handshakeTimer);
    const timeoutHandle = setTimeout(() => {
        const s = sessions.get(globalId);
        if (s && !s.isConnected && s.connectionState !== 'qr_ready') {
            console.log(`[${globalId}] ⚠️ Handshake Timeout (60s). Retrying...`);
            try { if (s.sock) s.sock.end(undefined); } catch (e) {}
            sessions.delete(globalId);
            setTimeout(() => startSession(userId, sessionId), 10000);
        }
    }, 60000);
    sessions.get(globalId).handshakeTimer = timeoutHandle;

    sock.ev.on('connection.update', (update) => {
        const { connection, lastDisconnect, qr } = update;
        const session = sessions.get(globalId);

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
                // Also clear the 60s handshake timer — otherwise it can fire AFTER this
                // close handler already scheduled a reconnect, delete the session mid-boot
                // and double-schedule startSession (the old "reconnect storm").
                if (session.handshakeTimer) clearTimeout(session.handshakeTimer);
            }

            const statusCode = (lastDisconnect?.error)?.output?.statusCode;
            const isLoggedOut = statusCode === DisconnectReason.loggedOut || statusCode === 401 || statusCode === 405;
            const shouldReconnect = !isLoggedOut;

            console.log(`[${globalId}] 🔌 Connection Closed. Status: ${statusCode}. Reconnect: ${shouldReconnect}`);

            if (isLoggedOut) {
                console.log(`[${globalId}] 🚨 SESSION INVALIDATED (Code: ${statusCode}). Cleaning up...`);
                if (session) {
                    if (session.handshakeTimer) clearTimeout(session.handshakeTimer);
                    if (session.sock) try { session.sock.end(undefined); } catch (e) {}
                }
                
                sessions.delete(globalId);
                io.emit('status', { sessionId, status: 'disconnected' });

                setTimeout(async () => {
                    console.log(`[${globalId}] 🗑️ Purging auth folder...`);
                    try {
                        fs.rmSync(authPath, { recursive: true, force: true });
                        console.log(`[${globalId}] ✅ Purge successful. Restarting...`);
                        startSession(userId, sessionId);
                    } catch (e) {
                        console.error(`[${globalId}] Purge failed:`, e.message);
                        startSession(userId, sessionId);
                    }
                }, 3000);
            } else if (shouldReconnect) {
                const attempts = (RECONNECT_ATTEMPTS.get(globalId) || 0) + 1;
                RECONNECT_ATTEMPTS.set(globalId, attempts);

                const delay = Math.min(3000 * Math.pow(2, attempts - 1), 120000);
                const status = attempts > 5 ? 'failure' : 'reconnecting';
                if (session) session.connectionState = status;
                io.emit('status', { sessionId, status });

                console.log(`[${globalId}] Reconnecting in ${delay/1000}s (Attempt ${attempts})...`);
                setTimeout(() => startSession(userId, sessionId), delay);
            } else {
                if (session && session.sock) session.sock.end(undefined);
                sessions.delete(globalId);
                io.emit('status', { sessionId, status: 'disconnected' });
            }
        } else if (connection === 'open') {
            RECONNECT_ATTEMPTS.set(globalId, 0); 
            if (session && session.handshakeTimer) clearTimeout(session.handshakeTimer);
            console.log(`[${globalId}] ✅ WhatsApp Connection Established.`);
            if (session) { session.isConnected = true; session.connectionState = 'connected'; }
            io.emit('status', { sessionId, status: 'connected' });

            if (session && session.watchdogTimer) clearInterval(session.watchdogTimer);
            const watchdog = setInterval(async () => {
                const s = sessions.get(globalId);
                if (!s || !s.isConnected || !s.sock) {
                    clearInterval(watchdog);
                    return;
                }
                try {
                    const pingPromise = s.sock.sendPresenceUpdate('available');
                    await Promise.race([
                        pingPromise,
                        new Promise((_, reject) => setTimeout(() => reject(new Error('Watchdog Ping Timeout: Socket Frozen')), 10000))
                    ]);
                } catch (e) {
                    console.error(`[${globalId}] 🧟 ZOMBIE CONNECTION DETECTED:`, e.message);
                    clearInterval(watchdog);
                    if (s.handshakeTimer) clearTimeout(s.handshakeTimer);
                    try { s.sock.end(undefined); } catch (err) {}
                    setTimeout(() => startSession(userId, sessionId), 2000);
                }
            }, 60000);
            if (session) session.watchdogTimer = watchdog;
        }
    });

    sock.ev.on('creds.update', saveCreds);

    sock.ev.on('contacts.upsert', (contacts) => {
        const session = sessions.get(globalId);
        if (!session) return;
        contacts.forEach(c => { if (c.name || c.notify) session.phonebook[c.id] = c.name || c.notify; });
    });

    sock.ev.on('messaging-history.set', async ({ messages }) => {
        io.emit('log', { sessionId, msg: `🔄 Decrypting & Saving History Sync to SQLite...` });
        let totalSaved = 0;
        for (const item of (messages || [])) {
            if (item.messages && Array.isArray(item.messages)) {
                for (const msg of item.messages) { if (await saveMessageToDB(globalId, msg, true)) totalSaved++; }
            } else if (item.key && item.message) {
                if (await saveMessageToDB(globalId, item, true)) totalSaved++;
            }
        }
        if (totalSaved > 0) io.emit('log', { sessionId, msg: `✅ DB DUMP COMPLETE: ${totalSaved} messages saved to SQLite.` });
        console.log(`[${globalId}] History Dump Complete: ${totalSaved} saved.`);
    });

    sock.ev.on('messages.upsert', async ({ messages, type }) => {
        console.log(`[${sessionId}] 📩 EVENT: messages.upsert | Type: ${type} | Count: ${messages.length}`);
        io.emit('log', { sessionId, msg: `DEBUG: UP_MSG Event (Type: ${type})` });

        if (type !== 'notify' && type !== 'append') {
            console.log(`[${sessionId}] ⏭ Skipping non-content update (Type: ${type})`);
            return;
        }

        for (const msg of messages) {
            try { // per-message isolation: one bad message must never kill the rest of the batch
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
                // LEARNING LOOP: if the OWNER (not the bot) typed this, and the bot had
                // recently replied in the same chat, treat it as a correction/example —
                // the human stepped in, so their words are ground truth for how this
                // conversation should have been handled. Fed back into future prompts.
                try {
                    const selfText = (savedData?.content?.text || '').trim();
                    const selfCleanId = savedData?.cleanId;
                    if (selfText && selfCleanId && !selfText.startsWith('[') && !isBotSentText(sessionId, selfCleanId, selfText)) {
                        const lastBot = await db.get(
                            `SELECT text, timestamp FROM messages WHERE session_id = ? AND contact_id = ? AND user_id = ?
                             AND message_id LIKE 'BOT_%' ORDER BY timestamp DESC LIMIT 1`,
                            [sessionId, selfCleanId, userId]);
                        if (lastBot && (Math.floor(Date.now() / 1000) - lastBot.timestamp) < 900) { // within 15 min of a bot reply
                            await db.run(`INSERT INTO reply_feedback (user_id, session_id, contact_id, bot_text, owner_text, created_at) VALUES (?,?,?,?,?,?)`,
                                [userId, sessionId, selfCleanId, (lastBot.text || '').slice(0, 400), selfText.slice(0, 400), Math.floor(Date.now() / 1000)]);
                            console.log(`[${sessionId}] 🧠 Learned from owner stepping in (${selfText.slice(0, 40)}…)`);
                        }
                    }
                } catch (e) { /* learning is best-effort, never blocks */ }
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
            const burstMode = await isBurstMode(userId, cleanId);
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
            const globalDoc = await db.get(`SELECT master_auto_reply, global_draft_mode FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);
            const masterEnabled = globalDoc ? (globalDoc.master_auto_reply === 1) : true;
            const globalDraftMode = globalDoc ? (globalDoc.global_draft_mode === 1) : false;

            const contactSettings = await db.get(`SELECT auto_reply, draft_mode FROM contacts WHERE session_id = ? AND contact_id = ? AND user_id = ?`, [sessionId, cleanId, userId]);
            const contactAutoReply = contactSettings ? (contactSettings.auto_reply === 1) : true;
            const contactDraftMode = contactSettings ? (contactSettings.draft_mode === 1) : false;
            
            const isDraftMode = globalDraftMode || contactDraftMode;

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
                let replyText = await generateSmartReply(userId, sessionId, contactName, cleanId, content.text);

                if (replyText) {
                    // INTERCEPT SCHEDULED MESSAGES
                    const scheduleRegex = /\[SCHEDULE_MESSAGE:\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^\]]+)\]/i;
                    const scheduleMatch = replyText.match(scheduleRegex);
                    if (scheduleMatch) {
                        try {
                            const dateStr = scheduleMatch[1].trim();
                            const targetStr = scheduleMatch[2].trim();
                            const schedMsg = scheduleMatch[3].trim();
                            
                            let targetContactId = cleanId;
                            let targetContactName = contactName;
                            
                            if (targetStr.toUpperCase() !== 'CURRENT') {
                                // Attempt fuzzy match for the target contact
                                const foundContact = await db.get(`SELECT contact_id, name FROM contacts WHERE session_id = ? AND user_id = ? AND name LIKE ? LIMIT 1`, 
                                    [sessionId, userId, `%${targetStr}%`]);
                                if (foundContact) {
                                    targetContactId = foundContact.contact_id;
                                    targetContactName = foundContact.name;
                                }
                            }
                            
                            const parsedMs = Date.parse(dateStr);
                            if (!isNaN(parsedMs)) {
                                const sendAt = Math.floor(parsedMs / 1000);
                                await db.run(`INSERT INTO scheduled_messages (user_id, session_id, contact_id, message, send_at_timestamp) VALUES (?, ?, ?, ?, ?)`,
                                    [userId, sessionId, targetContactId, schedMsg, sendAt]);
                                io.emit('log', { sessionId, msg: `⏰ Scheduled message for ${targetContactName} at ${dateStr}` });
                            }
                        } catch (e) {
                            console.error("Schedule Parse Error", e);
                        }
                        replyText = replyText.replace(scheduleRegex, '').trim();
                    }

                    // --- AGENT: LEAD CAPTURE ---
                    const leadRegex = /\[LEAD:\s*status\s*=\s*([^|\]]+?)\s*(?:\|\s*summary\s*=\s*([^\]]*?))?\]/i;
                    const leadMatch = replyText.match(leadRegex);
                    if (leadMatch) {
                        try {
                            const raw = (leadMatch[1] || 'interested').trim().toLowerCase();
                            const status = ['interested', 'hot', 'negotiating', 'won', 'lost'].includes(raw) ? raw : 'interested';
                            const summary = (leadMatch[2] || '').trim().slice(0, 200);
                            const nowTs = Math.floor(Date.now() / 1000);
                            await db.run(
                                `INSERT INTO leads (user_id, session_id, contact_id, contact_name, status, summary, created_at, updated_at)
                                 VALUES (?,?,?,?,?,?,?,?)
                                 ON CONFLICT(user_id, session_id, contact_id) DO UPDATE SET
                                   status=excluded.status, summary=excluded.summary, contact_name=excluded.contact_name, updated_at=excluded.updated_at`,
                                [userId, sessionId, cleanId, contactName, status, summary, nowTs, nowTs]
                            );
                            io.emit('lead_update', { sessionId, contactId: cleanId, contactName, status, summary });
                            io.emit('log', { sessionId, msg: `🎯 Lead: ${contactName} → ${status}${summary ? ` (${summary})` : ''}` });
                        } catch (e) { console.error('Lead capture error:', e.message); }
                        replyText = replyText.replace(leadRegex, '').trim();
                    }

                    // --- AGENT: HUMAN HANDOFF ---
                    const handoffRegex = /\[HANDOFF:\s*([^\]]+)\]/i;
                    const handoffMatch = replyText.match(handoffRegex);
                    if (handoffMatch) {
                        try {
                            const reason = (handoffMatch[1] || 'Needs owner attention').trim().slice(0, 300);
                            const nowTs = Math.floor(Date.now() / 1000);
                            await db.run(
                                `INSERT INTO notifications (user_id, session_id, contact_id, contact_name, type, message, created_at) VALUES (?,?,?,?,?,?,?)`,
                                [userId, sessionId, cleanId, contactName, 'handoff', reason, nowTs]
                            );
                            io.emit('notification', { sessionId, contactId: cleanId, contactName, type: 'handoff', message: reason, timestamp: Date.now() });
                            io.emit('log', { sessionId, msg: `🚨 HANDOFF needed: ${contactName} — ${reason}` });
                        } catch (e) { console.error('Handoff error:', e.message); }
                        replyText = replyText.replace(handoffRegex, '').trim();
                    }

                    if (!replyText || replyText.length === 0) continue;
                    if (isDraftMode) {
                        console.log(`[${sessionId}] 📝 DRAFT MODE ENABLED. Emitting draft instead of sending.`);
                        io.emit('new_draft', {
                            sessionId,
                            contactId: cleanId,
                            contactName,
                            userId,
                            jid,
                            originalMessage: content.text,
                            draftText: replyText,
                            timestamp: Date.now()
                        });
                        io.emit('log', { sessionId, msg: `📝 Draft generated for ${contactName} (waiting for approval)` });
                        continue; // Skip actual sending
                    }

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

                    // NORMAL TEXT REPLY — with human typing rhythm.
                    // A real person doesn't type a 3-line answer in 1.5s: scale the
                    // "typing…" window with reply length (~45ms/char) plus jitter,
                    // clamped so short replies feel snappy and long ones never stall.
                    await sock.sendPresenceUpdate('composing', jid);
                    const typingMs = Math.min(1200 + replyText.length * 45 + Math.random() * 900, 7000);
                    await delay(typingMs);
                    await sock.sendMessage(jid, { text: replyText });
                    noteBotSentText(sessionId, cleanId, replyText); // keep this out of style samples

                    // SAVE BOT REPLY
                    const botId = 'BOT_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
                    await db.run(`INSERT INTO messages (message_id, session_id, contact_id, user_id, text, sender, timestamp, date, is_from_me, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                        [botId, sessionId, cleanId, userId, replyText, MY_NAME, Math.floor(Date.now() / 1000), new Date().toISOString(), 1, null]);

                    // Trigger Graph Extraction asynchronously
                    extractKnowledgeTriples(userId, sessionId, cleanId).catch(err => console.error("Graph extraction failed:", err));

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
            } catch (msgErr) { // outer per-message guard (save/settings/burst phases)
                console.error(`[${sessionId}] 🔥 Message processing error (continuing with batch):`, msgErr.message);
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

    // --- ONE-DEVICE LINKING: pairing code instead of QR ---
    // Rides on the existing QR-stage socket (the exact state Baileys needs), so the
    // session lifecycle is untouched. User types the code into WhatsApp on the SAME
    // phone: Settings → Linked Devices → Link a Device → "Link with phone number instead".
    socket.on('request_pairing_code', async (data) => {
        if (!userId || !data?.sessionId || !data?.phone) return;
        const globalId = `${userId}_${data.sessionId}`;
        const session = sessions.get(globalId);
        const fail = (error) => socket.emit('pairing_code', { sessionId: data.sessionId, error });
        try {
            // Normalize: digits only; drop international 00 prefix; reject junk early.
            let phone = String(data.phone).replace(/[^0-9]/g, '');
            if (phone.startsWith('00')) phone = phone.slice(2);
            if (phone.length < 8 || phone.length > 15) {
                return fail('Enter your full number WITH country code (e.g. 2348031234567 — no + or spaces).');
            }
            if (phone.startsWith('0')) {
                return fail('That looks like a local number. Start with your country code (e.g. 234... not 0...).');
            }
            if (!session || !session.sock) {
                return fail('Node is not running yet. Start it first (the QR screen), then request a code.');
            }
            if (session.sock.authState?.creds?.registered) {
                return fail('This node is already linked to a WhatsApp account.');
            }
            // Rate-limit: one code per 20s per node (WhatsApp throttles aggressive requests).
            const now = Date.now();
            if (session.lastPairingReq && now - session.lastPairingReq < 20000) {
                return fail('Please wait a few seconds before requesting another code.');
            }
            session.lastPairingReq = now;

            const code = await session.sock.requestPairingCode(phone);
            const pretty = code && code.length === 8 ? `${code.slice(0, 4)}-${code.slice(4)}` : code;
            socket.emit('pairing_code', { sessionId: data.sessionId, code: pretty });
            io.emit('log', { sessionId: data.sessionId, msg: `🔗 Pairing code issued (${phone.slice(0, 5)}…)` });
        } catch (e) {
            console.error(`[${globalId}] Pairing code error:`, e.message);
            fail('Could not generate a code — make sure the node just started (QR visible), then try again.');
        }
    });

    socket.on('delete_session', (id) => {
        if (!id || !userId) return;
        const globalId = `${userId}_${id}`;
        const s = sessions.get(globalId); 
        if (s && s.sock) {
            try { s.sock.end(undefined); } catch (e) {}
        }
        sessions.delete(globalId);
        try { 
            const path = `./data/auth_baileys_${globalId}`;
            fs.rmSync(path, { recursive: true, force: true }); 
            console.log(`[SYSTEM] Deleted persistent session folder: ${path}`);
        } catch (e) { }
        socket.emit('session_deleted', id);
    });

    socket.on('get_config', async (sessionId) => {
        if (!userId) return;
        const config = await db.get(`SELECT * FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);
        socket.emit('config_data', { 
            sessionId, 
            config: {
                globalPrompt: config?.custom_prompt || '',
                masterAutoReply: config ? config.master_auto_reply === 1 : true,
                aiEngine: config?.active_api || 'deepseek',
                chakaModel: config?.chaka_model || 'chaka-medium'
            }
        });
    });

    socket.on('update_config', async (data) => {
        if (!userId || !data.updates) return;
        const { globalPrompt, masterAutoReply, aiEngine, chakaModel } = data.updates;
        
        const existing = await db.get(`SELECT id FROM global_settings WHERE id = 'settings' AND user_id = ?`, [userId]);
        if (!existing) {
            await db.run(`INSERT INTO global_settings (id, user_id, custom_prompt, master_auto_reply, active_api, chaka_model) VALUES ('settings', ?, ?, ?, ?, ?)`,
                [userId, globalPrompt || '', masterAutoReply === false ? 0 : 1, aiEngine || 'deepseek', chakaModel || 'chaka-medium']);
        } else {
            if (globalPrompt !== undefined) await db.run(`UPDATE global_settings SET custom_prompt = ? WHERE id = 'settings' AND user_id = ?`, [globalPrompt, userId]);
            if (masterAutoReply !== undefined) await db.run(`UPDATE global_settings SET master_auto_reply = ? WHERE id = 'settings' AND user_id = ?`, [masterAutoReply ? 1 : 0, userId]);
            if (aiEngine !== undefined) await db.run(`UPDATE global_settings SET active_api = ? WHERE id = 'settings' AND user_id = ?`, [aiEngine, userId]);
            if (chakaModel !== undefined) await db.run(`UPDATE global_settings SET chaka_model = ? WHERE id = 'settings' AND user_id = ?`, [chakaModel, userId]);
        }
        socket.emit('log', { sessionId: data.sessionId, msg: `⚙️ Global Config Updated` });
    });

    socket.on('get_contacts', async (sessionId) => {
        if (!userId) return;
        const rows = await db.all(`SELECT DISTINCT contact_id as id, name, jid FROM contacts WHERE user_id = ? AND session_id = ? ORDER BY last_active DESC LIMIT 100`, [userId, sessionId]);
        // Map to cleaner objects if needed, but UI seems happy with id/name/jid
        const contacts = rows.map(r => ({ ...r, cleanId: r.id }));
        socket.emit('contact_list', { sessionId, contacts });
    });

    socket.on('get_contact_settings', async (data) => {
        if (!userId || !data.contactId) return;
        const row = await db.get(`SELECT auto_reply, custom_prompt FROM contacts WHERE user_id = ? AND session_id = ? AND contact_id = ?`, [userId, data.sessionId, data.contactId]);
        socket.emit('contact_settings', {
            contactId: data.contactId,
            auto_reply: row ? row.auto_reply === 1 : true,
            custom_prompt: row?.custom_prompt || ''
        });
    });

    socket.on('update_contact_settings', async (data) => {
        if (!userId || !data.contactId || !data.settings) return;
        const { custom_prompt, auto_reply } = data.settings;
        await db.run(`UPDATE contacts SET custom_prompt = ?, auto_reply = ? WHERE user_id = ? AND session_id = ? AND contact_id = ?`,
            [custom_prompt || '', auto_reply ? 1 : 0, userId, data.sessionId, data.contactId]);
        socket.emit('log', { sessionId: data.sessionId, msg: `🎯 Entity Settings Updated` });
    });

    // --- AI keys are platform-hosted (env), not user-supplied ---
    // These handlers are retained as no-ops for backwards-compat with older clients;
    // the dashboard no longer exposes key management. They never store user keys.
    socket.on('list_api_keys', async () => {
        socket.emit('api_keys_list', []);
    });
    socket.on('add_api_key', async () => {
        socket.emit('api_keys_list', []);
    });
    socket.on('delete_api_key', async () => {
        socket.emit('api_keys_list', []);
    });

    socket.on('fetch_db_sessions', async () => {
        if (!userId) return;
        const rows = await db.all(`SELECT DISTINCT session_id FROM contacts WHERE user_id = ?`, [userId]);
        const dbSessions = rows.map(r => r.session_id);
        socket.emit('db_session_list', dbSessions);
    });

    socket.on('get_db_stats', async (sessionId) => {
        if (!userId) return;
        const msgCount = await db.get(`SELECT COUNT(*) as count FROM messages WHERE user_id = ? AND session_id = ?`, [userId, sessionId]);
        const stickerCount = await db.get(`SELECT COUNT(*) as count FROM stickers WHERE user_id = ? AND session_id = ?`, [userId, sessionId]);
        
        let sizeMb = 0;
        try {
            const stats = fs.statSync('./data/chaka_data.db');
            sizeMb = (stats.size / (1024 * 1024)).toFixed(2);
        } catch(e) {}

        socket.emit('db_stats', {
            sessionId,
            messageCount: msgCount?.count || 0,
            stickerCount: stickerCount?.count || 0,
            sizeMb
        });
    });

    socket.on('get_chat_history', async (data) => {
        if (!userId || !data.contactId) return;
        const rows = await db.all(`SELECT * FROM messages WHERE user_id = ? AND session_id = ? AND contact_id = ? ORDER BY timestamp DESC LIMIT 50`,
            [userId, data.sessionId, data.contactId]);
        socket.emit('chat_history', { sessionId: data.sessionId, contactId: data.contactId, messages: rows.reverse() });
    });

    socket.on('approve_draft', async (data) => {
        if (!userId || !data.sessionId || !data.jid || !data.draftText) return;
        const globalId = `${userId}_${data.sessionId}`;
        const session = sessions.get(globalId);
        if (session && session.sock) {
            try {
                await session.sock.sendPresenceUpdate('composing', data.jid);
                await delay(500);
                await session.sock.sendMessage(data.jid, { text: data.draftText });
                // Keep the echo of this send out of writing-style samples (it's AI text,
                // not the owner's own typing — even when owner-approved).
                noteBotSentText(data.sessionId, data.contactId, data.draftText);

                // GOLD-STANDARD LEARNING: the owner EDITED the AI's draft before sending.
                // That diff is the purest correction signal we get — store it.
                if (data.originalDraft && data.originalDraft.trim() !== data.draftText.trim()) {
                    await db.run(`INSERT INTO reply_feedback (user_id, session_id, contact_id, bot_text, owner_text, created_at) VALUES (?,?,?,?,?,?)`,
                        [userId, data.sessionId, data.contactId, data.originalDraft.slice(0, 400), data.draftText.slice(0, 400), Math.floor(Date.now() / 1000)]);
                    console.log(`[${data.sessionId}] 🧠 Learned from draft edit.`);
                }

                const botId = 'BOT_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
                await db.run(`INSERT INTO messages (message_id, session_id, contact_id, user_id, text, sender, timestamp, date, is_from_me, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                    [botId, data.sessionId, data.contactId, userId, data.draftText, MY_NAME, Math.floor(Date.now() / 1000), new Date().toISOString(), 1, null]);

                extractKnowledgeTriples(userId, data.sessionId, data.contactId).catch(err => console.error(err));
                io.emit('log', { sessionId: data.sessionId, msg: `🚀 Draft Sent to ${data.contactName}` });
            } catch (err) {
                console.error("Failed to send draft:", err);
            }
        }
    });

    socket.on('toggle_draft_mode', async (data) => {
        if (!userId) return;
        if (data.type === 'global') {
            await db.run(`UPDATE global_settings SET global_draft_mode = ? WHERE id = 'settings' AND user_id = ?`, [data.enabled ? 1 : 0, userId]);
        } else if (data.type === 'contact' && data.sessionId && data.contactId) {
            await db.run(`UPDATE contacts SET draft_mode = ? WHERE session_id = ? AND contact_id = ? AND user_id = ?`, [data.enabled ? 1 : 0, data.sessionId, data.contactId, userId]);
        }
    });
});

// --- BOOT SEQUENCE ---
async function bootServer() {
    const PORT = process.env.PORT || 3000;
// Fail fast on listen errors (e.g. port already in use). The global
// uncaughtException handler would otherwise swallow these and leave the
// process "running" without an HTTP listener.
server.on('error', (err) => {
    console.error(`💥 FATAL: HTTP server failed to start: ${err.message}`);
    process.exit(1);
});
server.listen(PORT, '0.0.0.0', async () => {
    console.log(`🚀 Server running and listening on http://0.0.0.0:${PORT}`);
    
    // --- AUTOMATIC DAILY BACKUPS (keep last 7) ---
    // VACUUM INTO writes a consistent snapshot even while the DB is live (WAL mode).
    const runBackup = async () => {
        try {
            if (!fs.existsSync('./data/backups')) fs.mkdirSync('./data/backups', { recursive: true });
            const stamp = new Date().toISOString().slice(0, 10);
            const dest = `./data/backups/chaka_data-${stamp}.db`;
            if (fs.existsSync(dest)) return; // one per day
            await db.exec(`VACUUM INTO '${dest}'`);
            const old = fs.readdirSync('./data/backups').filter(f => f.startsWith('chaka_data-')).sort();
            while (old.length > 7) { fs.unlinkSync(`./data/backups/${old.shift()}`); }
            console.log(`🗄️ Backup written: ${dest}`);
        } catch (e) { console.error('Backup failed:', e.message); }
    };
    setTimeout(runBackup, 60000);            // first backup a minute after boot
    setInterval(runBackup, 6 * 3600 * 1000); // check every 6h (no-ops if today's exists)

    // --- SELF-REVIEW SCHEDULER (learning flywheel) ---
    setTimeout(() => distillLessons().catch(() => {}), 120000);          // first pass 2 min after boot
    setInterval(() => distillLessons().catch(() => {}), 12 * 3600 * 1000); // then every 12h (skips users with no new feedback)

    // --- START CRON-BRAIN (PHASE 3) ---
    setInterval(async () => {
        try {
            const now = Math.floor(Date.now() / 1000);
            const dueMessages = await db.all(`SELECT * FROM scheduled_messages WHERE status = 'pending' AND send_at_timestamp <= ?`, [now]);
            for (const sched of dueMessages) {
                console.log(`[CRON] ⏰ Sending scheduled message to ${sched.contact_id}...`);
                const globalId = `${sched.user_id}_${sched.session_id}`;
                const session = sessions.get(globalId);
                if (session && session.sock) {
                    const jid = sched.contact_id.includes('@s.whatsapp.net') || sched.contact_id.includes('@g.us') || sched.contact_id.includes('@lid')
                                ? sched.contact_id 
                                : `${sched.contact_id}@s.whatsapp.net`;
                    await session.sock.sendMessage(jid, { text: sched.message });
                    noteBotSentText(sched.session_id, sched.contact_id, sched.message); // keep out of style samples

                    // Save as BOT_CRON
                    await db.run(`INSERT INTO messages (message_id, session_id, contact_id, user_id, text, sender, timestamp, date, is_from_me) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                        ['BOT_CRON_' + Date.now(), sched.session_id, sched.contact_id, sched.user_id, sched.message, MY_NAME, Math.floor(Date.now() / 1000), new Date().toISOString(), 1]);
                    
                    io.emit('log', { sessionId: sched.session_id, msg: `⏰ Scheduled message sent to ${sched.contact_id}!` });
                    await db.run(`UPDATE scheduled_messages SET status = 'sent' WHERE id = ?`, [sched.id]);
                } else {
                    console.error(`[CRON] Error: Session not found for ${globalId}`);
                }
            }
        } catch (e) {
            console.error("[CRON] Error:", e.message);
        }
    }, 60000); // Check every 60 seconds
});

    await initDB();
    await initLocalAI();

    // UNIFIED SESSION RECOVERY & SCRUBBER
    if (fs.existsSync('./data')) {
        const folders = fs.readdirSync('./data').filter(f => f.startsWith('auth_baileys_'));
        console.log(`[SYSTEM] Recovering ${folders.length} accounts from persistent disk...`);
        for (const f of folders) {
            const globalId = f.replace('auth_baileys_', '');

            // Proactive Scrubbing for corrupted session IDs
            if (globalId.includes('_undefined') || globalId.endsWith('_')) {
                 console.log(`[SCRUBBER] Deleting corrupted session folder from disk: ${f}`);
                 try { fs.rmSync(`./data/${f}`, { recursive: true, force: true }); } catch (e) {}
                 continue;
            }

            const parts = globalId.split('_');
            if (parts.length >= 2) {
                const userId = parts[0];
                const id = parts.slice(1).join('_');
                if (userId && id && !sessions.has(globalId)) {
                    console.log(`[BOOT] Waking node: ${id} for user: ${userId}`);
                    startSession(userId, id);
                    await new Promise(r => setTimeout(r, 8000)); // Increased stagger for safety
                }
            }
        }
    }
}

// --- GRACEFUL SHUTDOWN ---
// Fly.io (and most orchestrators) send SIGTERM before stopping a machine. Close the
// HTTP server and the database cleanly so in-flight requests finish and SQLite is not
// left mid-write. A hard timeout guards against hanging connections.
let shuttingDown = false;
async function gracefulShutdown(signal) {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`\n🛑 Received ${signal} — shutting down gracefully...`);

    const forceExit = setTimeout(() => {
        console.error("⏱️ Shutdown timed out, forcing exit.");
        process.exit(1);
    }, 15000);
    forceExit.unref();

    try {
        io.close();
        await new Promise((resolve) => server.close(resolve));
        if (db) await db.close();
        console.log("✅ Clean shutdown complete.");
        clearTimeout(forceExit);
        process.exit(0);
    } catch (e) {
        console.error("💥 Error during shutdown:", e);
        process.exit(1);
    }
}
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

bootServer();
