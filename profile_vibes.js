
import admin from 'firebase-admin';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const serviceAccount = require('./serviceAccountKey.json');

if (!admin.apps.length) {
    admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
}
const db = admin.firestore();

// --- CONFIG ---
const MY_NAME = 'Temple';
const ANALYSIS_DEPTH = 200; // Fetch more to ensure we get enough of OUR messages

// --- SCORING FUNCTIONS ---
function getAvgSentenceLength(messages) {
    if (!messages.length) return 0;
    const totalWords = messages.reduce((acc, msg) => {
        if (!msg.text) return acc;
        return acc + msg.text.split(' ').length;
    }, 0);
    return Math.round(totalWords / messages.length);
}

function getTopEmojis(messages) {
    const emojiMap = {};
    const emojiRegex = /[\p{Emoji_Presentation}\p{Extended_Pictographic}]/gu;

    messages.forEach(msg => {
        if (!msg.text) return;
        const matches = msg.text.match(emojiRegex);
        if (matches) {
            matches.forEach(e => {
                emojiMap[e] = (emojiMap[e] || 0) + 1;
            });
        }
    });

    return Object.entries(emojiMap)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(e => e[0]);
}

function getMediaUsage(messages) {
    let audio = 0;
    let sticker = 0;
    let image = 0;
    let video = 0;

    messages.forEach(msg => {
        if (!msg.text) return;
        if (msg.text.includes('[AUDIO_MESSAGE]')) audio++;
        if (msg.text.includes('[STICKER]')) sticker++;
        if (msg.text.includes('[IMAGE]')) image++;
        if (msg.text.includes('[VIDEO]')) video++;
    });

    return { audio, sticker, image, video };
}

function getToneIndicators(messages) {
    let casual = 0;
    let formal = 0;

    messages.forEach(msg => {
        if (!msg.text) return;
        const lower = msg.text.toLowerCase();
        if (lower.includes('lol') || lower.includes('haha') || lower.includes('lmao') || lower.includes('u ') || lower.includes('ur ')) {
            casual++;
        }
        if (msg.text.endsWith('.') || msg.text.endsWith('?')) {
            formal++;
        }
    });

    if (casual > formal) return "Casual/Slang";
    if (formal > casual * 2) return "Formal";
    return "Neutral";
}


// --- MAIN LOOP ---

async function runProfiler() {
    console.log("🚀 Starting Vibe Profiler...");

    const contactsRef = db.collection('whatsapp_data');
    const snapshot = await contactsRef.listDocuments();

    if (snapshot.length === 0) {
        console.log("No contacts found in database.");
        return;
    }

    console.log(`Found ${snapshot.length} contacts to analyze.`);

    for (const docRef of snapshot) {
        const contactId = docRef.id;
        process.stdout.write(`Analyzing ${contactId}... `);

        // SIMPLE QUERY: Just get recent messages (Avoid Index Error)
        // We will filter for 'Temple' in Javascript
        const msgsSnapshot = await docRef.collection('messages')
            .orderBy('timestamp', 'desc')
            .limit(ANALYSIS_DEPTH)
            .get();

        if (msgsSnapshot.empty) {
            console.log(`Skipped (No messages)`);
            continue;
        }

        // IN-MEMORY FILTER
        const allMessages = msgsSnapshot.docs.map(d => d.data());
        const myMessages = allMessages.filter(m => m.sender === MY_NAME);

        if (myMessages.length === 0) {
            console.log(`Skipped (No messages from you)`);
            continue;
        }

        // --- ANALYSIS ---
        const profile = {
            last_updated: new Date().toISOString(),
            sample_size: myMessages.length,
            avg_words_per_msg: getAvgSentenceLength(myMessages),
            top_emojis: getTopEmojis(myMessages),
            media_habits: getMediaUsage(myMessages),
            detected_tone: getToneIndicators(myMessages),
        };

        // --- SAVE ---
        await docRef.set({
            vibe_profile: profile
        }, { merge: true });

        console.log(`✅ Saved! Tone: ${profile.detected_tone}`);
    }

    console.log("\n🎉 Profiling Complete!");
}

runProfiler();
