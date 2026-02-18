import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import makeWASocket, { useMultiFileAuthState, DisconnectReason, fetchLatestBaileysVersion } from '@whiskeysockets/baileys';
import qrcode from 'qrcode';
import admin from 'firebase-admin';
import { createRequire } from 'module';
import fs from 'fs';

// --- SETUP ---
const require = createRequire(import.meta.url);

let serviceAccount;
try {
    if (process.env.FIREBASE_CREDENTIALS) {
        serviceAccount = JSON.parse(process.env.FIREBASE_CREDENTIALS);
    } else {
        serviceAccount = require('./serviceAccountKey.json');
    }
} catch (error) {
    console.error("Failed to load Firebase credentials:", error);
    process.exit(1);
}

if (!admin.apps.length) {
    admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
}
const db = admin.firestore();

const app = express();
const server = createServer(app);
const io = new Server(server);

app.use(express.static('public'));

// --- GLOBAL PHONEBOOK ---
let phonebook = {};

// --- CUSTOM MEMORY STORE ---
const store = {
    messages: {},
    bind: (ev) => {
        ev.on('messages.upsert', ({ messages }) => {
            for (const msg of messages) {
                const jid = msg.key.remoteJid;
                if (!store.messages[jid]) store.messages[jid] = [];
                if (!store.messages[jid].find(m => m.key.id === msg.key.id)) {
                    store.messages[jid].push(msg);
                }
                if (store.messages[jid].length > 1000) {
                    store.messages[jid] = store.messages[jid].slice(-1000);
                }
            }
        });
    },
    loadMessage: async (jid, id) => {
        const chat = store.messages[jid];
        return chat?.find(m => m.key.id === id);
    }
};

// --- SMART TEXT EXTRACTOR (Context-Aware Version) ---
function extractContent(msg) {
    if (!msg) return { type: 'unknown', text: null };

    // 1. Direct Text (Priority)
    const text = msg.conversation ||
        msg.extendedTextMessage?.text ||
        msg.imageMessage?.caption ||
        msg.videoMessage?.caption;

    if (text) return { type: 'text', text: text };

    // 2. Context Markers (Crucial for AI Personality)
    // Helps the AI understand "I sent a voice note here" or "I reacted with a sticker"
    if (msg.audioMessage) return { type: 'text', text: "[AUDIO_MESSAGE]" };
    if (msg.stickerMessage) return { type: 'text', text: "[STICKER]" };
    if (msg.imageMessage) return { type: 'text', text: "[IMAGE]" };
    if (msg.videoMessage) return { type: 'text', text: "[VIDEO]" };
    if (msg.contactMessage) return { type: 'text', text: "[CONTACT_CARD]" };
    if (msg.locationMessage) return { type: 'text', text: "[LOCATION]" };

    // 3. Deep Dive (Recursion for Wrappers)
    const deepMsg = msg.viewOnceMessage?.message ||
        msg.viewOnceMessageV2?.message ||
        msg.ephemeralMessage?.message ||
        msg.documentWithCaptionMessage?.message;

    if (deepMsg) return extractContent(deepMsg);

    return { type: 'junk', text: null };
}

// --- HELPER: SAFE TIMESTAMP ---
function getSafeTimestamp(ts) {
    if (typeof ts === 'number') return ts;
    if (ts && typeof ts.low === 'number') return ts.low;
    return Math.floor(Date.now() / 1000);
}

// --- MAIN BOT ---
async function startSock() {
    const { state, saveCreds } = await useMultiFileAuthState('auth_baileys');
    const { version } = await fetchLatestBaileysVersion();

    console.log(`Using WA v${version.join('.')}`);

    const sock = makeWASocket({
        version,
        auth: state,
        printQRInTerminal: true,
        syncFullHistory: true,
        getMessage: async (key) => {
            if (store) {
                const msg = await store.loadMessage(key.remoteJid, key.id);
                return msg?.message || undefined;
            }
            return { conversation: 'hello' };
        }
    });

    store.bind(sock.ev);

    sock.ev.on('connection.update', (update) => {
        const { connection, lastDisconnect, qr } = update;

        if (qr) {
            qrcode.toDataURL(qr, (err, url) => {
                io.emit('qr_code', url);
                io.emit('log', 'Please scan the QR code...');
            });
        }

        if (connection === 'close') {
            const shouldReconnect = (lastDisconnect?.error)?.output?.statusCode !== DisconnectReason.loggedOut;
            io.emit('log', `⚠️ Connection closed. Reconnecting...`);
            if (shouldReconnect) startSock();
        } else if (connection === 'open') {
            io.emit('status', 'connected');
            io.emit('log', '✅ CONNECTED! Waiting for Contacts & Chats...');
        }
    });

    sock.ev.on('creds.update', saveCreds);

    // --- 1. CONTACTS HANDLER ---
    sock.ev.on('contacts.upsert', (contacts) => {
        contacts.forEach(c => {
            if (c.name || c.notify) {
                phonebook[c.id] = c.name || c.notify;
            }
        });
        io.emit('log', `📚 Phonebook updated: ${Object.keys(phonebook).length} contacts found.`);
    });

    // --- 2. HISTORY HANDLER ---
    sock.ev.on('messaging-history.set', async ({ messages, contacts }) => {
        if (contacts) {
            contacts.forEach(c => {
                phonebook[c.id] = c.name || c.notify || c.id.split('@')[0];
            });
        }

        const total = messages.length;
        if (total === 0) return;

        io.emit('log', `🔥 HISTORY DUMP: Processing ${total} items...`);

        let savedCount = 0;
        let skippedCount = 0;

        const batch = db.batch();
        let batchOpCount = 0;

        for (const item of messages) {
            try {
                const msg = item.message ? item : item.key ? item : null;
                if (!msg) continue;

                const jid = msg.key.remoteJid;

                if (jid === 'status@broadcast') {
                    skippedCount++;
                    continue;
                }

                const content = extractContent(msg.message);
                if (!content.text) {
                    skippedCount++;
                    continue;
                }

                const contactName = phonebook[jid] || msg.pushName || jid.split('@')[0];
                const cleanId = jid.replace(/[^a-zA-Z0-9]/g, "_");

                const safeTs = getSafeTimestamp(msg.messageTimestamp);
                const safeDate = new Date(safeTs * 1000).toISOString();

                const docRef = db.collection('whatsapp_data')
                    .doc(cleanId)
                    .collection('messages')
                    .doc(msg.key.id);

                batch.set(docRef, {
                    text: content.text,
                    sender: msg.key.fromMe ? 'Temple' : contactName,
                    timestamp: safeTs,
                    date: safeDate,
                });

                batchOpCount++;
                savedCount++;

                if (savedCount % 10 === 0) {
                    io.emit('processing_contact', { name: contactName, id: cleanId });
                    io.emit('contact_update', { name: contactName, id: cleanId, count: savedCount });
                }

                if (batchOpCount >= 400) {
                    await batch.commit();
                    batchOpCount = 0;
                    io.emit('log', `...saved ${savedCount} messages`);
                }
            } catch (err) {
                console.error("Error processing msg:", err);
            }
        }

        if (batchOpCount > 0) await batch.commit();

        io.emit('log', `✅ DONE: ${savedCount} saved. (${skippedCount} system msgs skipped)`);
    });

    // --- 3. LIVE HANDLER ---
    sock.ev.on('messages.upsert', async ({ messages, type }) => {
        if (type !== 'notify' && type !== 'append') return;

        for (const msg of messages) {
            try {
                const jid = msg.key.remoteJid;
                if (jid === 'status@broadcast') continue;

                const content = extractContent(msg.message);
                if (!content.text) continue;

                const contactName = phonebook[jid] || msg.pushName || jid.split('@')[0];
                const cleanId = jid.replace(/[^a-zA-Z0-9]/g, "_");

                const safeTs = getSafeTimestamp(msg.messageTimestamp);

                await db.collection('whatsapp_data').doc(cleanId).collection('messages').doc(msg.key.id).set({
                    text: content.text,
                    sender: msg.key.fromMe ? 'Temple' : contactName,
                    timestamp: safeTs,
                    date: new Date(safeTs * 1000).toISOString()
                });

                io.emit('processing_contact', { name: contactName, id: cleanId });
                io.emit('log', `✅ New msg from ${contactName}`);
            } catch (e) {
                console.error("Live save error:", e);
            }
        }
    });
}

io.on('connection', (socket) => {
    socket.emit('log', 'System Ready. Initializing...');
});

startSock();

server.listen(3000, () => {
    console.log('Server running on http://localhost:3000');
});
