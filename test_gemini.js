import { GoogleGenerativeAI } from "@google/generative-ai";

const key = "AIzaSyBfDH-F0jlDsduSqF2rNpZjSJbyPIZxX1o";
const genAI = new GoogleGenerativeAI(key);

async function test() {
    const models = ["gemini-3.1-flash-lite-preview", "gemini-2.5-flash", "gemini-2.0-flash"];
    for (const m of models) {
        try {
            console.log(`Testing ${m}...`);
            const model = genAI.getGenerativeModel({ model: m });
            const result = await model.generateContent("hi");
            console.log(`✅ ${m} works! Response: ${result.response.text()}`);
        } catch (e) {
            console.log(`❌ ${m} failed: ${e.message}`);
        }
    }
}

test();
