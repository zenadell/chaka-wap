const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const admin = require('firebase-admin');
const serviceAccount = require('./serviceAccountKey.json'); // GET THIS FROM FIREBASE CONSOLE

// 1. Initialize Firebase
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount)
});
const db = admin.firestore();

// 2. Initialize the Crawler (Headless Chrome)
const client = new Client({
    authStrategy: new LocalAuth(), // Saves login so you don't scan QR every time
    puppeteer: {
        headless: false, // Set to true if you don't want to see the browser pop up
    }
});

// 3. Generate QR Code
client.on('qr', (qr) => {
    console.log('SCAN THIS QR CODE WITH YOUR ANDROID PHONE:');
    qrcode.generate(qr, { small: true });
});

client.on('ready', async () => {
    console.log('✅ Client is ready! Starting the crawler...');
    
    // 4. Get All Chats
    const chats = await client.getChats();
    console.log(`Found ${chats.length} chats.`);

    for (const chat of chats) {
        if (chat.isGroup) continue; // Skip groups if you only want personal training data

        const contact = await chat.getContact();
        const contactName = contact.name || contact.pushname || chat.name;
        
        // Clean name for Firebase ID (remove weird chars)
        const cleanName = contactName.replace(/[^a-zA-Z0-9 ]/g, "").trim() || "Unknown";
        
        console.log(`Crawler crawling: ${cleanName}...`);

        // 5. Fetch History (Limit: Infinity grabs everything loaded on Web)
        // Note: Web version might not have YEARS of history synced immediately.
        let messages = [];
        try {
            messages = await chat.fetchMessages({ limit: 500 }); // Adjust limit as needed
        } catch (e) {
            console.log(`Error fetching ${cleanName}:`, e.message);
            continue;
        }

        if (messages.length === 0) continue;

        // 6. Prepare Batch for Firebase
        const batch = db.batch();
        let operationCount = 0;

        for (const msg of messages) {
            if (!msg.body) continue; // Skip media/stickers

            const docRef = db.collection('dataset_whatsapp')
                .doc(cleanName)
                .collection('history')
                .doc(msg.id.id); // Use WhatsApp ID to prevent duplicates

            batch.set(docRef, {
                text: msg.body,
                sender: msg.fromMe ? "Temple" : cleanName,
                timestamp: msg.timestamp,
                isReply: msg.hasQuotedMsg,
                type: msg.type
            });
            
            operationCount++;
        }

        await batch.commit();
        console.log(`Saved ${operationCount} messages for ${cleanName}`);
    }

    console.log("🔥 CRAWL COMPLETE. All data is in Firebase.");
    process.exit();
});

client.initialize();
