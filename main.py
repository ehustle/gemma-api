from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI()

# Load Gemma 3 (1B) instruction-tuned model
pipe = pipeline(
    "text-generation",
    model="google/gemma-3-1b-it",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

class ChatRequest(BaseModel):
    user_prompt: str

@app.post("/generate")
async def generate_text(data: ChatRequest):
    messages = [[
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": data.user_prompt}]
        },
    ]]
    output = pipe(messages, max_new_tokens=150)
    return {"result": output[0]["generated_text"]}
