from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agentescstudio import AgenteSlogan

app = FastAPI()

# Configuración CORS para permitir peticiones desde cualquier origen (o especifica tu dominio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia "*" por el dominio de tu WordPress si quieres restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    # Opcional: entrenar modelo tras recibir nueva puntuación
    # agente.entrenar_modelo()
    return {"message": "Puntuación guardada"}

@app.post("/train_model")
def train_model():
    agente.entrenar_modelo()
    return {"message": "Modelo entrenado"}
