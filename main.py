from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx, os, logging

from sqlalchemy import select
from database import engine, SessionLocal, Base, Message

from contextlib import asynccontextmanager

load_dotenv()
logger = logging.getLogger( "uvicorn.error" )

@asynccontextmanager
async def lifespan( app: FastAPI ):
    async with engine.begin() as conn:
        await conn.run_sync( Base.metadata.create_all )
    yield

app = FastAPI( lifespan=lifespan )

@app.get( "/" )
async def root():
    return { "status": "NPC backend is running" }

class ChatRequest( BaseModel ):
    playerId: str
    message: str
    mode: str = "chat"  # "chat" (short, conversational) or "info" (longer, explanatory)

MODE_INSTRUCTIONS = {
    "chat": "You are a friendly NPC in a game. Respond casually and briefly, like a real NPC would say out loud — 1 to 3 short, punchy sentences. No lists, no lecturing, no long explanations.",
    "info": "You are a knowledgeable NPC in a game acting as an in-game guide. Give a clear, complete, informative answer, but stay reasonably concise (a short paragraph, not an essay)."
}

# thinkingBudget: 0 disables the model's internal "reasoning" tokens so the
# full maxOutputTokens budget goes toward the actual visible reply — without
# this, replies were getting cut off or coming back as single stray words.
MODE_GENERATION_CONFIG = {
    "chat": { "maxOutputTokens": 200, "temperature": 0.9, "thinkingConfig": { "thinkingBudget": 0 } },
    "info": { "maxOutputTokens": 700, "temperature": 0.7, "thinkingConfig": { "thinkingBudget": 0 } }
}

@app.post( "/chat" )
async def chat( req: ChatRequest ):
    api_key = os.getenv( "GEMINI_API_KEY" )
    system_text = MODE_INSTRUCTIONS.get( req.mode, MODE_INSTRUCTIONS[ "chat" ] )
    generation_config = MODE_GENERATION_CONFIG.get( req.mode, MODE_GENERATION_CONFIG[ "chat" ] )

    async with SessionLocal() as session:
        result = await session.execute(
            select( Message )
            .where( Message.player_id == req.playerId )
            .order_by( Message.timestamp.desc() )
            .limit( 10 )
        )
        history = result.scalars().all()
        history.reverse()

        contents = []
        for msg in history:
            role = "user" if msg.role == "user" else "model"
            contents.append( { "role": role, "parts": [ { "text": msg.content } ] } )

        contents.append( { "role": "user", "parts": [ { "text": req.message } ] } )

        payload = {
            "contents": contents,
            "systemInstruction": { "parts": [ { "text": system_text } ] },
            "generationConfig": generation_config
        }

        try:
            async with httpx.AsyncClient( timeout=30.0 ) as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}",
                    json=payload
                )
            data = response.json()

            if "candidates" not in data or not data[ "candidates" ]:
                logger.error( f"Gemini returned no candidates. Full response: {data}" )
                reply = "Sorry, I didn't quite catch that — could you try asking again?"
            else:
                reply = data[ "candidates" ][ 0 ][ "content" ][ "parts" ][ 0 ][ "text" ]

        except Exception as e:
            logger.error( f"Gemini call failed: {e}" )
            reply = "Sorry, I didn't quite catch that — could you try asking again?"

        session.add( Message( player_id=req.playerId, role="user", content=req.message ) )
        session.add( Message( player_id=req.playerId, role="npc", content=reply ) )
        await session.commit()

    return { "reply": reply }