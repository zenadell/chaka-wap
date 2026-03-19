import os
import subprocess
import sys
import time

def run_command(command):
    print(f"Executing: {command}")
    # Force output to be immediate for Colab logs
    sys.stdout.flush()
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        print(f"❌ Command failed with return code {process.returncode}")
        return False
    return True

print("🦍 =================================================")
print("🦍 QWEN-7B COLAB GPU ENGINE (BULLETPROOF VERSION)")
print("🦍 =================================================")
print("\n[1/4] Installing dependencies (This takes 5-10 mins for GPU drivers)...")

# 1. Install dependencies with CUDA
# We use --force-reinstall to make sure the CUDA version is the one active
if not run_command("CMAKE_ARGS='-DGGML_CUDA=on' pip install --force-reinstall llama-cpp-python --no-cache-dir"):
    print("FATAL: Failed to install llama-cpp-python with CUDA.")
    sys.exit(1)

run_command("pip install fastapi uvicorn pydantic nest_asyncio pyngrok huggingface_hub")

# 2. Setup FastAPI Server
print("\n[2/4] Preparing the AI server script...")
server_code = """
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nest_asyncio
import uvicorn
from llama_cpp import Llama
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download

nest_asyncio.apply()

MODEL_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
MODEL_FILE = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

llm = None

def init_model():
    global llm
    
    # 🦍 SELF-HEALING: Purge old Qwen files if they exist to save space
    for old_file in ["Qwen2.5-7B-Instruct-Q4_K_M.gguf"]:
        if os.path.exists(old_file):
            print(f"🧹 Purging old model file: {old_file}")
            os.remove(old_file)

    if os.path.exists(MODEL_FILE):
        if os.path.getsize(MODEL_FILE) < 1000000:
            print(f"⚠️ Detected corrupted model file (too small). Purging...")
            os.remove(MODEL_FILE)

    if not os.path.exists(MODEL_FILE):
        print(f"📥 Downloading {MODEL_FILE} from Hugging Face...")
        try:
            hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=".", local_dir_use_symlinks=False)
            print("✅ Download Complete.")
        except Exception as e:
            print(f"❌ Download Failed: {e}")
            sys.exit(1)
    
    print("⚙️ Loading model into GPU (Offloading all layers)...")
    try:
        llm = Llama(
            model_path=MODEL_FILE,
            n_gpu_layers=-1, # Force ALL to GPU
            n_ctx=2048,
            verbose=False
        )
        print("🚀 MODEL LOADED SUCCESSFULLY ON GPU!")
    except Exception as e:
        print(f"❌ Model Load Failed: {e}")
        sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_model()
    yield

app = FastAPI(title="Colab Qwen API", lifespan=lifespan)

class ChatRequest(BaseModel):
    system_prompt: str = "You are a helpful assistant."
    user_prompt: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if llm is None:
        raise HTTPException(status_code=500, detail="Server Error: Model not initialized.")
    
    try:
        # Structured chat completion for better instruction following
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt}
            ],
            max_tokens=1024,
            temperature=0.8, # Slightly higher for more variety
            top_p=0.95
        )
        return {"response": response["choices"][0]["message"]["content"].strip()}
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    print("📡 Starting AI engine with high-capacity timeout settings...")
    # Crank up the timeouts for large prompt processing
    uvicorn.run(
        app, 
        host='0.0.0.0', 
        port=8000, 
        timeout_keep_alive=600,
        limit_concurrency=10
    )
"""

with open("colab_api.py", "w") as f:
    f.write(server_code)

# 3. Setup ngrok
print("\n[3/4] Establishing Ngrok Tunnel...")
NGROK_AUTHTOKEN = "3Atnc6YdKVSiErxf69spWIVqmlH_Tdi5z9kccbXTwFJpMpCi" 

if not NGROK_AUTHTOKEN:
    print("❌ ERROR: No NGROK_AUTHTOKEN found in script.")
    sys.exit(1)

run_command(f"ngrok config add-authtoken {NGROK_AUTHTOKEN}")
from pyngrok import ngrok
try:
    public_url = ngrok.connect(8000).public_url
    print(f"\n🔥 YOUR PUBLIC ENDPOINT IS: {public_url}/api/chat")
    print("1. Copy the URL above.")
    print("2. Paste it into your Mac's .env as QWEN_ENDPOINT.")
    print("3. Restart your node server.\n")
except Exception as e:
    print(f"❌ Ngrok Tunnel Failed: {e}")
    sys.exit(1)

# 4. Start Server
print("[4/4] Booting AI Engine...")
run_command("python colab_api.py")
