from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx, os

from sqlalchemy import select
from database import engine, SessionLocal, Base, Message

from contextlib import asynccontextmanager

load_dotenv()       # reads .env file into environment variables

@asynccontextmanager    # create tables on startup
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync( Base.metadata.create_all )
    yield       # app runs
                # nothing needed for shutdown
app = FastAPI( lifespan = lifespan )

@app.get("/")
async def root():
    return {"status": "NPC backend is running"}

class ChatRequest( BaseModel ):
    playerId: str
    message: str

@app.post( "/chat" )
async def chat( req: ChatRequest ):
    api_key = os.getenv( "GEMINI_API_KEY" )
    async with SessionLocal() as session:
        # pull last 10 messages for this player, by most recent
        result = await session.execute(
            select( Message )
            .where( Message.player_id == req.playerId )
            .order_by( Message.timestamp.desc() )
            .limit(10)
        )
        history = result.scalars().all()
        history.reverse()   # put back in chronological order

        # build Gemini's 'contents' list from history
        contents = []
        for msg in history:
            role = "user" if msg.role == "user" else "model"
            contents.append( {"role": role, "parts": [{"text":msg.content}]})

        # add incoming message
        contents.append({"role": "user", "parts": [{"text": req.message}]})

        # call Gemini (full context)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}",
                json = { "contents": contents } 
            )
            data = response.json()
            reply = data["candidates"][0]["content"]["parts"][0]["text"]

        # log both player's message and NPC's reply
        session.add( Message(player_id = req.playerId, role = "user", content = req.message) )
        session.add( Message(player_id=req.playerId, role="npc", content=reply) )
        await session.commit()

    return { "reply": reply }



