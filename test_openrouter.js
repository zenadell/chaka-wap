import 'dotenv/config';

async function testOpenRouter() {
    const key = process.env.OPENROUTER_API_KEY || "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";
    console.log("Testing OpenRouter via native fetch with key starting with:", key.substring(0, 10));

    try {
        const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${key}`,
                'Content-Type': 'application/json',
                'HTTP-Referer': 'http://localhost:3000',
                'X-OpenRouter-Title': 'WhatsApp Crawler AI'
            },
            body: JSON.stringify({
                model: 'qwen/qwen3-coder:free',
                messages: [
                    { role: 'user', content: 'Say exactly: "OpenRouter via Fetch operates flawlessly!"' }
                ]
            })
        });

        const data = await response.json();
        
        if (data && data.choices && data.choices.length > 0) {
            console.log("✅ SUCCESS:");
            console.log(data.choices[0].message.content.trim());
        } else {
            console.log("❌ FAILED: Unexpected response format.");
            console.log(JSON.stringify(data, null, 2));
        }
    } catch (error) {
        console.error("🔥 Error:", error.message);
    }
}

testOpenRouter();
