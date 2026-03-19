import { GoogleGenerativeAI } from "@google/generative-ai";
const key = "AIzaSyBfDH-F0jlDsduSqF2rNpZjSJbyPIZxX1o";
const genAI = new GoogleGenerativeAI(key);

async function list() {
    try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${key}`;
        const response = await fetch(url);
        const data = await response.json();
        console.log(JSON.stringify(data, null, 2));
    } catch (e) {
        console.log(e);
    }
}
list();
