import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from llama_cpp import Llama

# Load environment variables
load_dotenv()

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/Users/mac/Desktop/orpheus-tts/models/qwen2.5-coder-32b-instruct-q4_k_m.gguf")

# Initialize FastAPI app
app = FastAPI(title="Local Qwen-32B API")

# Global variables for the model
llm = None

def init_model():
    global llm
    if llm is None:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model not found at {MODEL_PATH}")
            return False
            
        print(f"Loading model from {MODEL_PATH}...")
        try:
            llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,  # Offload all to Metal
                n_ctx=2048,       # Context window
                n_batch=512,      # Batch size
                verbose=False
            )
            print("Model loaded successfully with Metal support!")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    return True

# Request model
class ChatRequest(BaseModel):
    message: str
    model: str = "qwen"

@app.on_event("startup")
async def startup_event():
    init_model()

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if llm is None:
        success = init_model()
        if not success:
            raise HTTPException(status_code=500, detail="Qwen model is not loaded or missing.")
            
    print(f"Received prompt: {request.message[:100]}...")
    
    try:
        # Note: llama-cpp-python has a create_chat_completion method
        # We need to adapt the prompt to Qwen's chatml format
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.message}
            ],
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9
        )
        
        reply = response["choices"][0]["message"]["content"].strip()
        print(f"Generated reply: {reply[:100]}...")
        
        return {"response": reply}
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🤖 QWEN-32B LOCAL ENGINE")
    print("="*50)
    print(f"Using model: {MODEL_PATH}")
    print("IMPORTANT: Ensure no other heavy AI apps are running.")
    print("Starting server on http://localhost:8000")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
