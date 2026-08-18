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
    mode: str = "chat"

MODE_INSTRUCTIONS = {
    "chat": "You are a friendly NPC in a game. Respond casually and conversationally, like a real NPC would say out loud — a few short sentences is fine, just don't lecture or write an essay. Always finish your thought completely; never trail off mid-sentence.",
    "info": "You are a knowledgeable NPC in a game acting as an in-game guide. Give a clear, complete, informative answer, but stay reasonably concise (a short paragraph, not an essay)."
}

MODE_GENERATION_CONFIG = {
    "chat": { "maxOutputTokens": 600, "temperature": 0.9 },
    "info": { "maxOutputTokens": 1000, "temperature": 0.7 }
}

GEMINI_MODEL = "gemini-3.6-flash"

QUOTA_FALLBACK_REPLY = "Whew — quite the crowd of question-askers today. My scrying glass needs a moment to recharge, traveler. Try again shortly!"

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

        reply = "Sorry, I didn't quite catch that — could you try asking again?"

        if not api_key:
            reply = "[DEBUG] GEMINI_API_KEY is missing/empty on the server."
        else:
            try:
                async with httpx.AsyncClient( timeout=30.0 ) as client:
                    response = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}",
                        json=payload
                    )

                if response.status_code == 429:
                    logger.error( f"Gemini quota exceeded: {response.text}" )
                    reply = QUOTA_FALLBACK_REPLY
                elif response.status_code != 200:
                    error_snippet = response.text[ :200 ]
                    logger.error( f"Gemini returned HTTP {response.status_code}: {response.text}" )
                    reply = f"[DEBUG] Gemini HTTP {response.status_code}: {error_snippet}"
                else:
                    data = response.json()
                    if "candidates" not in data or not data[ "candidates" ]:
                        logger.error( f"Gemini returned no candidates. Full response: {data}" )
                        reply = f"[DEBUG] No candidates returned: {str(data)[:200]}"
                    else:
                        candidate = data[ "candidates" ][ 0 ]
                        finish_reason = candidate.get( "finishReason", "" )
                        if finish_reason == "MAX_TOKENS":
                            logger.warning( "Gemini hit MAX_TOKENS — reply may be truncated despite raised limit." )
                        reply = candidate[ "content" ][ "parts" ][ 0 ][ "text" ]

            except Exception as e:
                logger.error( f"Gemini call failed: {e}" )
                reply = f"[DEBUG] Exception: {type(e).__name__}: {str(e)[:150]}"

        session.add( Message( player_id=req.playerId, role="user", content=req.message ) )
        session.add( Message( player_id=req.playerId, role="npc", content=reply ) )
        await session.commit()

    return { "reply": reply }