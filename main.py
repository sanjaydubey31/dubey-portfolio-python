from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
from fastapi.middleware.cors import CORSMiddleware
import re
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langChain_chat_model import get_chat_response

load_dotenv()

LANGFLOW_TOKEN = os.getenv("LANGFLOW_TOKEN")

app = FastAPI()

# Allow frontend on port 5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://main.d2raojdfdmfx91.amplifyapp.com/"],  # Or ["*"] for all origins (not recommended for prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/langflow")
async def run_langflow(request: Request):
    payload = await request.json()
    print("Received payload:", payload)
    headers = {
        "Authorization": f"Bearer {LANGFLOW_TOKEN}",  # replace with real token
        "Content-Type": "application/json"
    }

    try:
        response = httpx.post(
            "https://api.langflow.astra.datastax.com/lf/8a23ea9d-49c2-4ab8-b760-4b6243ebc692/api/v1/run/ee1b615b-d9e8-42a4-833c-4bb0752da9e3?stream=false",  # replace UUID/FLOW_ID
            json=payload,
            headers=headers,
            timeout=100.0
        )
        
        if response.status_code == 200:
            return JSONResponse(content=response.json())
        else:
            return JSONResponse(status_code=response.status_code, content={
                "error": "Langflow API returned an error",
                "status_code": response.status_code,
                "text": response.text
            })

    except httpx.RequestError as exc:
        return JSONResponse(status_code=500, content={
            "error": "Request to Langflow failed",
            "details": str(exc)
        })


class ChatRequest(BaseModel):
    input_value: str

@app.post("/langChain")
async def chat_endpoint(chat_req: ChatRequest):
    reply = get_chat_response(chat_req.input_value)
    return {"response": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8013, reload=True)
