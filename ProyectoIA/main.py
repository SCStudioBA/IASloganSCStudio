from fastapi import FastAPI
from pydantic import BaseModel
from agentescstudio import AgenteSlogan

app = FastAPI()

class SloganRequest(BaseModel):
    rubro: str
    modo: str

class RatingRequest(BaseModel):
    slogan: str
    rubro: str
    modo: str
    puntuacion: int

agente = AgenteSlogan()

@app.post("/generate_slogan")
def generate_slogan(data: SloganRequest):
    candidatos = agente.generar_varios_slogans(data.rubro, data.modo, cantidad=3)
    mejor_slogan = agente.elegir_mejor_slogan(candidatos, umbral_minimo=2.5)
    return {"slogan": mejor_slogan}

@app.post("/submit_rating")
def submit_rating(data: RatingRequest):
    agente.guardar_puntuacion(data.slogan, data.rubro, data.modo, data.puntuacion)
    return {"message": "Puntuación guardada"}

@app.post("/train_model")
def train_model():
    agente.entrenar_modelo()
    return {"message": "Modelo entrenado"}
