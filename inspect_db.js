import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import fs from 'fs';

async function inspect() {
    console.log("🔍 Inspecting SQLite Database...");

    if (!fs.existsSync('chaka_data.db')) {
        console.error("❌ Database file 'chaka_data.db' not found.");
        return;
    }

    const db = await open({
        filename: 'chaka_data.db',
        driver: sqlite3.Database
    });

    const msgCount = await db.get('SELECT COUNT(*) as count FROM messages');
    const contactCount = await db.get('SELECT COUNT(*) as count FROM contacts');
    const lastMsgs = await db.all('SELECT sender, text, date FROM messages ORDER BY timestamp DESC LIMIT 5');
    const settings = await db.get('SELECT * FROM global_settings');

    console.log("\n--- STATISTICS ---");
    console.log(`Total Messages: ${msgCount.count}`);
    console.log(`Total Contacts: ${contactCount.count}`);
    console.log(`Database Size: ${(fs.statSync('chaka_data.db').size / 1024).toFixed(2)} KB`);

    console.log("\n--- RECENT ACTIVITY ---");
    lastMsgs.forEach(m => {
        console.log(`[${m.date}] ${m.sender}: ${m.text.substring(0, 50)}...`);
    });

    console.log("\n--- GLOBAL SETTINGS ---");
    console.log(`Master Auto-Reply: ${settings.master_auto_reply === 1 ? 'ON' : 'OFF'}`);
    console.log(`Custom Prompt: ${settings.custom_prompt || 'None'}`);

    await db.close();
}

inspect();
