from fastapi import FastAPI, Depends, status
from fastapi.exceptions import HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound
from bot import AnswerBot
from config import settings
from db import schemas, models, engine, get_db
from typing import Union, List

models.Base.metadata.create_all(bind=engine)

app = FastAPI()
bot = AnswerBot()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
api_key = '3ef78c01c63c9834b505fe22877e2b5a23bc0fcc624e53c7377bd652164cc216'


def api_key_auth(key: str = Depends(oauth2_scheme)):
    if key not in api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )

@app.post("/pergunta_db", response_model=schemas.Resposta, dependencies=[Depends(api_key_auth)])
async def api_bot(pergunta: schemas.PerguntaDB):
    resposta, score, start, end, contexto = bot.pergunta(pergunta.Pergunta, IdContexto=pergunta.IdContexto)

    return schemas.Resposta(Resposta=resposta, Score=score, Start=start, End=end, Contexto=contexto)

@app.post("/pergunta_contexto", response_model=schemas.Resposta, dependencies=[Depends(api_key_auth)])
async def api_bot(pergunta: schemas.PerguntaContexto):
    resposta, score, start, end, contexto = bot.pergunta(pergunta.Pergunta, IdContexto=0, contexto=pergunta.Contexto)

    return schemas.Resposta(Resposta=resposta, Score=score, Start=start, End=end, Contexto=contexto)


@app.get("/lista_contexto", response_model=List[schemas.ListaTitulos])
async def api_get_lista_contexto(db: Session = Depends(get_db)):

    return bot.get_lista_contextos(db)


@app.post("/contexto", response_model=schemas.ContextoPerguntas)
async def api_get_contexto(contexto: schemas.GetContexto, db: Session = Depends(get_db)):
    try:
        model = schemas.Contexto.from_orm(bot.get_contexto(contexto.IdContexto, db))
        return schemas.ContextoPerguntas(**model.dict(), Perguntas=bot.get_perguntas(contexto.IdContexto, db))
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Contexto não encontrado")


@app.post("/add_contexto", response_model=schemas.Contexto, dependencies=[Depends(api_key_auth)])
async def api_add_contexto(contexto: schemas.AddContexto, db: Session = Depends(get_db)):

    return bot.add_contexto(db=db, contexto=contexto)

@app.post("/del_contexto", response_model=schemas.confimation, dependencies=[Depends(api_key_auth)])
async def api_del_contexto(contexto: schemas.GetContexto, db: Session = Depends(get_db)):

    return bot.del_contexto(db=db, id=contexto.IdContexto)

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = form_data.username
    password = form_data.password
    if user != "botce" or password != "12345":
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"access_token": api_key, "token_type": "bearer"}




DESC = """

![logo](https://www.tcerj.tc.br/cdn-storage/logos/logo-reduzida-colorida_full-para_fundo_branco.svg)

API de acesso ao Question Answering Bot do TCE-RJ.

"""

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=settings.TITLE,
        description=DESC,
        version=settings.VERSION,
        routes=app.routes,
    )

    openapi_schema["info"]["x-logo"] = {
        "url": settings.LOGO_TCE
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


#
app.openapi_schema = custom_openapi()

