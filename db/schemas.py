from typing import Union, List
from pydantic import BaseModel


class PerguntaDB(BaseModel):
    IdContexto: int
    Pergunta: str

class PerguntaContexto(BaseModel):
    Contexto: str
    Pergunta: str


class Resposta(BaseModel):
    Resposta: str
    Score: float
    Start: int
    End: int
    Contexto: str


class GetContexto(BaseModel):
    IdContexto: int

class ListaTitulos(BaseModel):
    IdContexto: int
    Titulo: str

    class Config:
        orm_mode = True

class Contexto(BaseModel):
    IdContexto: int
    Contexto: str
    Titulo: str

    class Config:
        orm_mode = True

class Perguntas(BaseModel):
    IdPergunta: int
    Pergunta: str


    class Config:
        orm_mode = True


class ContextoPerguntas(Contexto):
    Perguntas: List[Perguntas]

    class Config:
        orm_mode = True


class AddContexto(BaseModel):
    Contexto: str
    Titulo: str

class confimation(BaseModel):
    operation: bool




