from transformers import pipeline
from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound
import io
from db import get_db, schemas
from db.models import DimContexto, DimPerguntas
from fastapi.exceptions import HTTPException


class AnswerBot:

    def __init__(self):

        model_name = 'pierreguillou/bert-base-cased-squad-v1.1-portuguese'
        self.nlp = pipeline("question-answering", model=model_name)
        self.db = next(get_db())
        self.model_coxt = DimContexto
        self.model_perg = DimPerguntas

    def get_lista_contextos(self, db: Session):
        return db.query(self.model_coxt.IdContexto, self.model_coxt.Titulo).all()

    def get_contexto(self, id: int, db: Session):
        return db.query(self.model_coxt).filter(self.model_coxt.IdContexto == id).one()

    def get_perguntas(self, id:int, db: Session):
        return db.query(self.model_perg).filter(self.model_perg.IdContexto == id).all()

    def add_contexto(self, contexto: schemas.AddContexto, db: Session):
        db_contexto = self.model_coxt(Contexto=contexto.Contexto, Titulo=contexto.Titulo)
        db.add(db_contexto)
        db.commit()
        db.refresh(db_contexto)
        return db_contexto

    def del_contexto(self, id: int, db: Session):
        db_contexto = db.query(self.model_coxt).filter(self.model_coxt.IdContexto == id)
        db.delete(db_contexto)
        db.commit()
        return {"operation": True}

    def pergunta(self, pergunta: str, IdContexto: int, contexto: str = "") -> tuple[str, float, int, int, str]:
        try:
            if IdContexto:
                query = self.get_contexto(id=IdContexto, db=self.db)
                contexto = query.Contexto

            resultado = self.nlp(question=pergunta, context=contexto)

            resposta = resultado['answer'].capitalize()
            score = float(round(resultado['score'], 4))
            start = resultado['start']
            end = resultado['end']

            print(f"Question: '{pergunta}'")
            print(f"Answer: '{resposta}', score: {score}, start: {start}, end: {end}")

        except (NoResultFound, ValueError) as e:
            raise HTTPException(status_code=404, detail="Contexto não encontrado") from e

        return resposta, score, start, end, contexto
