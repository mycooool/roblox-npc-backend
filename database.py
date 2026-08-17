from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, Integer, Text, DateTime, func
import os

from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv( "DATABASE_URL" )
engine = create_async_engine( DATABASE_URL, pool_pre_ping = True, pool_recycle = 300 )
SessionLocal = sessionmaker( engine, class_ = AsyncSession, expire_on_commit = False )
Base = declarative_base()

class Message( Base ):
    __tablename__ = "messages"
    id = Column( Integer, primary_key = True )
    player_id = Column( String, index = True )
    role = Column( String )
    content = Column( Text )
    timestamp = Column( DateTime, server_default = func.now() )