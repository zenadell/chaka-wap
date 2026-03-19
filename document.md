# Chaka AI Engine: Comprehensive Project Documentation
**Version:** 1.0 (Phase 48 Architecture)
**Lead Developer/Trainer:** Jomiez (Under the leadership of Templeton, Founder/CEO of Chaka-AI)

---

## 🏗️ 1. Project Overview & Architecture
Chaka is a highly advanced, multi-modal, local-first AI engine designed to operate autonomously as a human-like participant on WhatsApp. Unlike traditional chatbots that use rigid menus or sound like customer service agents, Chaka is built on a "Dynamic Style Learning Engine" and a "Roadman" persona prompt, enabling it to aggressively mirror the conversational style (slang, brevity, spelling) of its interaction partner. 

The system operates via a decoupled architecture:
1. **WhatsApp Web Bridge:** A Headless WhatsApp Web client (`@whiskeysockets/baileys`) linking a phone number to the node.
2. **SQLite Memory Engine:** A highly robust local database that stores an encrypted, persistent log of all chat history, contacts, and stickers.
3. **Local AI Vector Store (RAG):** An intelligent associative memory system utilizing local transformers (`@xenova/transformers`) to embed messages, allowing the AI to recall facts from months ago.
4. **AI Orchestrator (Multi-LLM):** A failover-protected routing layer that queries advanced LLMs (primarily Gemini 2.5 Flash and Chaka's custom endpoints) to generate responses.
5. **Real-Time Telemetry Dashboard (Frontend):** A React-inspired (Vanilla JS + TailwindCSS) command center visualizing the AI's internal thoughts, vector indexing, and database health.

---

## 🛠️ 2. Tech Stack & Libraries Use

**Backend (Core Engine):**
- **Node.js**: The runtime environment.
- **`@whiskeysockets/baileys`**: The core library used to reverse-engineer the WhatsApp Web socket protocol. It allows the script to log in via QR code, listen to messages, download media, and send replies.
- **`sqlite3` & `sqlite`**: Used for the primary relational database (`chaka_data.db`). It stores `messages`, `contacts`, `stickers`, and `global_settings`. SQLite was chosen for its ultra-fast local read/write speeds, crucial for real-time chat logging and vector storage.
- **`@google/generative-ai`**: The official SDK to interact with Gemini models (specifically `gemini-2.5-flash` for the main brain and `gemini-1.5-flash` for Vision capabilities).
- **`@xenova/transformers`**: A crucial library that runs Hugging Face models *locally* in Node.js. We use `all-MiniLM-L6-v2` to generate dense vector embeddings (Float32Arrays) of text. Running this locally avoids API costs for memory generation and ensures absolute privacy.
- **`express` & `socket.io`**: Used to serve the real-time admin dashboard. Socket.io pushes live logs, AI thought processes, and database stats to the frontend with zero latency.
- **`qrcode` & `qrcode-terminal`**: Used for generating the WhatsApp pairing code in both the terminal and the web UI.

**Frontend (Admin Dashboard):**
- **HTML5/Vanilla JS**: No heavy frontend frameworks were used to ensure the dashboard remains incredibly lightweight.
- **TailwindCSS**: Supplied via a CDN for rapid, modern, responsive styling (glassmorphism, dark hero themes).
- **GSAP (GreenSock)**: Used for high-end micro-animations and smooth transitions in the UI.

---

## 🧠 3. Memory Logic & Retrieval-Augmented Generation (RAG)
Chaka doesn't just read the last message; it remembers the entire relationship.

### **The Ingestion Pipeline:**
1. A message arrives.
2. If it contains text, it is passed to the local `@xenova/transformers` model (`MiniLM-L6-v2`).
3. The model converts the text into a 384-dimensional mathematical vector (an embedding).
4. The message text, sender data, and the *binary BLOB* of this vector are saved to SQLite.

### **The "Deep RAG" Retrieval Pipeline:**
When a user asks a question (e.g., "What did we talk about last month regarding the project?"):
1. The AI embeds the *question* into a vector.
2. The engine performs a **Cosine Similarity Search** against the entire SQLite database to find the top 3 mathematically closest messages.
3. These retrieved historical facts are injected into the hidden "System Prompt" before the AI sees the message.
4. *Result:* The AI answers with knowledge of the past, without explicitly saying "I searched my database."

---

## 🎭 4. Response Method & Style Learning
Chaka is designed to never sound like an AI. 

### **The Anti-Bot Prompt (Roadman Vibe):**
The core system prompt (`generatesmartreply`) strictly forbids phrases like "How can I help you?", "As an AI", or the use of perfect punctuation. It relies on a "Roadman" base vocabulary (using words like *typeshit*, *fvck*, *bro*, *chilling*). 

### **The Dynamic Style Learning Engine (Phase 46):**
Instead of relying solely on the hardcoded prompt, Chaka actively learns the user's style:
1. Before replying, it queries the database for the last 5-8 messages sent *by the specific human user*.
2. It analyzes their syntax: Do they use capital letters? Do they send 2-word replies? Do they use Nigerian Pidgin?
3. This analysis is passed to Gemini as a strict rule: "You must mirror this exact style."
4. If the chat is brand new or quiet, the system uses a "Vibe Fallback," looking at how Templeton generally talks to maintain a cohesive personality.

---

## 👁️ 5. Vision Engine & Contextual Occasion Analytics (Phase 48)
Chaka is fully multimodal. It doesn't just read text; it actually looks at pictures.

### **How it works:**
1. When a user sends an image, Baileys intercepts the encrypted media buffer.
2. Before analyzing the image, Chaka pulls the last 5 chat messages to understand the *current conversational context*.
3. The image buffer and the chat context are sent to `gemini-1.5-flash` (used specifically for its binary inline stability).
4. **Occasion Inference:** The AI doesn't just say "This is a photo of a burger." It says, "The user sent a picture of a burger because they mentioned they were hungry 2 minutes ago."
5. This translated description/reasoning is injected into the chat history as `[IMAGE: ... REASON: ...]`, allowing the main text AI (`gemini-2.5-flash`) to casually comment on the user's photo.

---

## 🔄 6. Failover Orchestrator (Phase 44)
WhatsApp bots frequently crash due to API rate limits (HTTP 429). Chaka is built with an "Ironclad Routing Orchestrator".

1. **Model Rotation:** If the primary Chaka endpoint fails, it cascades through backup models (`Velocity`, `Core`, `Reasoning`, `Ultra`).
2. **Provider Failover:** If all Chaka specific endpoints are down, it seamlessly redirects the traffic directly to Google Gemini.
3. **Key Rotation:** It iterates through an array of injected Gemini API keys (`API_KEYS`). If one key exhausts its quota, it switches to the next instantly.
4. **Burst Protection:** If messages arrive too fast (spam), the `BURST_TRACKER` temporarily throttles AI context depth to protect token limits (TPM).

---

## 🛑 7. Past Failures & Replacements
Throughout development, several technologies and approaches were attempted and ultimately replaced for better performance:

1. **Failure: Firebase Cloud Firestore**
   - *Attempt:* Initially, we tried to store chat history and vectors in Firebase for cloud accessibility.
   - *Reason for Failure:* Firestore is too slow for real-time dense vector retrieval and charges heavily for read/writes. Embedding arrays also exceeded document size limits.
   - *Replacement:* **Local SQLite**. It provides instantaneous reads, zero cost, and seamless BLOB storage for binary vectors.

2. **Failure: OpenAI/Gemini Embeddings (`text-embedding-004`)**
   - *Attempt:* Using cloud APIs to generate the mathematical vectors for memory.
   - *Reason for Failure:* It burned through API credits incredibly fast, especially during the "Deep Scrape" phase (indexing 20,000+ historical messages). It also triggered rate limits constantly.
   - *Replacement:* **`@xenova/transformers` (`all-MiniLM-L6-v2`)**. Running the embedding model locally inside Node.js is 100% free, entirely private, and bypassed all rate limits.

3. **Failure: Float Array Database Storage**
   - *Attempt:* Storing vectors as comma-separated `.toString()` strings in the database.
   - *Reason for Failure:* String parsing took too much CPU time and bloated the database size dramatically.
   - *Replacement:* **Binary BLOBs**. Vectors are now converted to `Float32Array`, wrapped in `Buffer`, and saved as raw BLOBs, reducing database size by thousands of megabytes and speeding up read times by 10x.

4. **Failure: Fixed Width UI Layouts**
   - *Attempt:* Using rigid `w-[360px]` Tailwind classes for the sidebar and persona controls.
   - *Reason for Failure:* The UI bled and broke aggressively on tablet screens (iPad views).
   - *Replacement:* **Flex Fluidity**. Replaced with `min-w-0`, `flex-wrap`, and `overflow-hidden` shields, allowing the dashboard to scale flawlessly from massive 4k monitors down to tiny phone screens.

---

## 🚀 8. Planned Future Upgrades
- **Phase 49: Audio Sync Protocol:** Implementing Google Cloud Speech-to-Text or local Whisper whisper-node implementations to allow the AI to "listen" to voice notes, and using Gemini's new Voice capabilities to stream audio replies directly back into WhatsApp.
