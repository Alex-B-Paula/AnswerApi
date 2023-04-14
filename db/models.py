from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Float, TIMESTAMP
from sqlalchemy.orm import relationship
from config import settings
from db import Base, engine


class DimContexto(Base):
    __tablename__ = "DimContexto"
    __table_args__ = {"schema": settings.schema}

    IdContexto = Column(Integer, primary_key=True, index=True)
    Contexto = Column(String)
    Titulo = Column(String(50))


class DimPerguntas(Base):
    __tablename__ = "DimPerguntas"
    __table_args__ = {"schema": settings.schema}

    IdPergunta = Column(Integer, primary_key=True, index=True)
    Pergunta = Column(String)
    IdContexto = Column(Integer, ForeignKey(DimContexto.IdContexto))

