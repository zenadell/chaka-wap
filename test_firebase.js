
import admin from 'firebase-admin';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const serviceAccount = require('./serviceAccountKey.json');

console.log("Initializing Firebase...");
if (!admin.apps.length) {
    admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
}
const db = admin.firestore();

async function testWrite() {
    try {
        console.log("Attempting to write to 'whatsapp_debug' collection...");
        const res = await db.collection('whatsapp_debug').add({
            timestamp: new Date().toISOString(),
            message: "Test write from Antigravity Agent",
            status: "success"
        });
        console.log("✅ Write Successful! Document ID:", res.id);

        console.log("Attempting to read it back...");
        const doc = await res.get();
        console.log("✅ Read Successful! Data:", doc.data());

        console.log("Cleaning up...");
        await res.delete();
        console.log("✅ Delete Successful!");

        console.log("🎉 FIREBASE IS FULLY FUNCTIONAL.");
    } catch (error) {
        console.error("❌ FIREBASE ERROR:", error);
    }
}

testWrite();
