import random
import nltk
from nltk.corpus import wordnet
from pytrends.request import TrendReq
import spacy
from gensim.models import Word2Vec
import time
import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import unicodedata
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

MODOS = ["popular", "semantico", "diferenciado"]

# --- Configuración inicial ---
nlp = spacy.load("es_core_news_sm")

corpus = [
    # Panadería (ampliado)
    [
        "pan", "masa", "levadura", "harina", "croissant", "baguette", "facturas", "tostadas", "bollos",
        "bizcocho", "horno", "artesanal", "tradición", "receta", "dulce", "salado",
        "crujiente", "esponjoso", "fresco", "natural", "sabroso", "casero", "integral", "glaseado",
        "miga", "fermentación", "repostería", "molde", "panadería", "panes",
        "masa madre", "panecillo", "tierno", "delicia", "horneado", "sabores", "corteza",
        "gluten", "semillas", "baguette", "pan dulce", "pan integral", "panadería artesanal",
        "panadería tradicional", "panadería casera", "panadería gourmet"
    ],

    # Mascotas (ampliado)
    [
        "perro", "gato", "mascota", "hamster", "conejo", "pez", "cuidado", "alimentación", "juguetes",
        "veterinario", "compañía", "amor", "felicidad", "entrenamiento", "adopción",
        "paseo", "adiestramiento", "salud", "vacunas", "higiene", "correa", "comida", "pelaje",
        "jaula", "accesorios", "cachorro", "adulto", "juguetón", "amistad", "protección", "bienestar",
        "collar", "cama", "comportamiento", "alimentación balanceada", "adiestrador", "pelota",
        "snacks", "baño", "cepillo", "adiestramiento positivo", "adiestramiento canino"
    ],

    # Gimnasio (ampliado)
    [
        "gym", "fuerza", "salud", "energía", "pesas", "cardio", "entrenamiento", "rutina", "músculo",
        "resistencia", "bienestar", "motivación", "crossfit", "yoga", "pilates",
        "flexibilidad", "tonificación", "pesas libres", "pesas rusas", "cinta", "abdominales",
        "calentamiento", "enfriamiento", "deporte", "atleta", "rendimiento", "saludable",
        "nutrición", "descanso", "superación", "objetivos", "entrenador", "clases grupales",
        "entrenamiento funcional", "cardiovascular", "pesas olímpicas", "rutina personalizada",
        "entrenamiento de fuerza", "ejercicio", "fitness", "bienestar integral"
    ],

    # Floristería (ampliado)
    [
        "flor", "florería", "floristería", "arreglo", "bouquet", "rosas", "tulipanes", "margaritas",
        "naturaleza", "colores", "fragancia", "ornamentación", "decoración", "evento", "regalo",
        "centro de mesa", "ramo", "jardinería", "plantas", "follaje", "florista", "floral",
        "temporada", "exótico", "silvestre", "fresco", "perfume", "boda", "celebración", "ambientación",
        "orquídeas", "lirios", "claveles", "flores frescas", "flores secas", "arte floral",
        "decoración de eventos", "paisajismo", "plantas ornamentales", "flores naturales"
    ],

    # Almacén (ampliado)
    [
        "almacén", "tienda", "mercado", "abarrotes", "productos", "comida", "bebida", "frutas",
        "verduras", "carnes", "lácteos", "hogar", "vecindario", "tradición", "diario",
        "ofertas", "precios", "frescura", "calidad", "atención", "cliente", "supermercado",
        "panadería", "charcutería", "conservas", "limpieza", "bebidas", "snacks", "orgánico",
        "local", "servicio", "variedad", "productos frescos", "productos locales", "productos importados",
        "productos enlatados", "productos congelados", "productos naturales", "productos artesanales"
    ],

    # Clases particulares (ampliado)
    [
        "clases", "particulares", "enseñanza", "aprendizaje", "profesor", "alumno", "educación",
        "matemáticas", "idiomas", "apoyo", "refuerzo", "conocimiento", "estudio", "motivación", "éxito",
        "tutoría", "exámenes", "preparación", "didáctico", "personalizado", "habilidades", "práctica",
        "teoría", "evaluación", "curso", "formación", "desarrollo", "competencias", "orientación",
        "asesoría", "clase online", "presencial", "educación a distancia", "educación virtual",
        "capacitación", "aprendizaje continuo", "técnicas de estudio", "planificación", "metodología"
    ]
]

modelo_path = "modelo_word2vec.model"

if os.path.exists(modelo_path):
    print("Cargando modelo Word2Vec guardado...")
    w2v_model = Word2Vec.load(modelo_path)
else:
    print("Entrenando modelo Word2Vec desde cero...")
    w2v_model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )
    w2v_model.save(modelo_path)
    print("Modelo guardado para futuras ejecuciones.")

def normalizar(texto):
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto

class AgenteSlogan:
    def __init__(self):
        self.puntuaciones_file = "puntuaciones.json"
        self.modelo_file = "modelo.pkl"
        self.vectorizer_file = "vectorizer.pkl"
        self.modelo = None
        self.vectorizer = None

        self.diferenciados = {
            "panaderia": [
                "Con lo que tenés y mi edición alcanza",
                "Pan con historia, hecho para vos",
                "Sabores que despiertan tu día",
                "Tradición horneada con cariño",
                "El aroma que abraza tu mañana",
                "Cada bocado, un recuerdo familiar",
                "Pan artesanal que enamora sentidos",
                "Del horno a tu mesa con amor",
                "Sabor que une generaciones",
                "Horneando sonrisas desde siempre",
                "Tu panadería de confianza y calidad",
                "Pasión y tradición en cada hogaza",
                "El secreto está en la masa madre",
                "Más que pan, momentos para compartir",
                "El sabor que te hace volver"
            ],
            "petshop": [
                "Tu mascota, nuestra inspiración",
                "Diseñamos su felicidad",
                "Cuidado y amor en cada detalle",
                "Más que mascotas, familia",
                "Donde el amor por ellos es primero",
                "Todo para su bienestar y alegría",
                "Juguetes, salud y cariño en un solo lugar",
                "Tu compañero merece lo mejor",
                "Cuidamos a quienes te hacen feliz",
                "Expertos en mimar a tus mascotas",
                "Porque ellos también son parte de casa",
                "Calidad y cariño para tu amigo fiel",
                "Tu petshop de confianza y dedicación",
                "Amor y cuidado en cada producto",
                "Hacemos felices a tus peludos"
            ],
            "gym": [
                "Entrenamiento con propósito",
                "Tu bienestar, nuestra misión",
                "Fuerza y disciplina para tu meta",
                "Superación en cada rutina",
                "Impulsa tu energía, transforma tu vida",
                "Donde el esfuerzo se convierte en éxito",
                "Más que un gimnasio, tu segundo hogar",
                "Motivación constante para tus objetivos",
                "Entrena duro, vive mejor",
                "Tu cuerpo, tu templo, tu gimnasio",
                "Potencia tu rendimiento cada día",
                "Entrenamiento personalizado para vos",
                "Supera tus límites con nosotros",
                "Fuerza, salud y pasión en cada sesión",
                "Transforma tu cuerpo, cambia tu vida"
            ],
            "floristeria": [
                "Dulces amapolas",
                "Lindas flores, grandes canciones",
                "Colores que emocionan",
                "Belleza natural en cada ramo",
                "Flores que hablan por vos",
                "El arte de regalar emociones",
                "Cada flor, un mensaje de amor",
                "Decoramos tus momentos especiales",
                "Fragancias que enamoran sentidos",
                "Ramos que cuentan historias",
                "La naturaleza en su máxima expresión",
                "Flores frescas para cada ocasión",
                "Detalles que hacen la diferencia",
                "Tu floristería de confianza",
                "Colores y aromas que inspiran"
            ],
            "almacen": [
                "Tu tienda de confianza",
                "Productos frescos, todos los días",
                "Cercanía que alimenta",
                "El sabor del barrio",
                "Calidad y servicio en cada compra",
                "Donde la tradición se encuentra con la frescura",
                "Variedad y precios que te sorprenden",
                "Tu mercado de confianza y cercanía",
                "Productos seleccionados para vos",
                "El alma del barrio en cada góndola",
                "Siempre cerca, siempre fresco",
                "Tu almacén, tu hogar",
                "Atención personalizada para vos",
                "Lo mejor para tu mesa y tu familia",
                "Confianza y calidad que perduran"
            ],
            "clases particulares": [
                "Aprendizaje a tu medida",
                "Tu éxito, nuestra misión",
                "Conocimiento que transforma",
                "Motivación para cada logro",
                "Educación personalizada para vos",
                "Potenciamos tu talento y esfuerzo",
                "Clases que hacen la diferencia",
                "Tu camino al éxito académico",
                "Aprende, crece y supera tus metas",
                "Apoyo educativo con pasión y dedicación",
                "Formamos futuros brillantes",
                "El acompañamiento que necesitás",
                "Metodologías innovadoras para vos",
                "Tu progreso, nuestra prioridad",
                "Más que clases, inspiración"
            ]
        }

        self.diccionario_local = {
            "panaderia": [
                "pan", "bollería", "repostería", "horneado", "masa", "croissant", "baguette",
                "bizcocho", "tostadas", "facturas", "dulce", "salado", "artesanal", "casero",
                "crujiente", "esponjoso", "fresco", "natural", "sabroso", "tradición", "receta",
                "levadura", "harina", "glaseado", "miga", "fermentación", "molde", "panecillo",
                "integral", "gluten", "masa madre", "pan dulce", "pan integral", "horno",
                "delicia", "corteza", "tierno", "sabores", "panadería artesanal", "panadería tradicional"
            ],
            "petshop": [
                "perro", "gato", "mascota", "hamster", "conejo", "pez", "cuidado", "alimentación",
                "juguetes", "veterinario", "compañía", "amor", "felicidad", "entrenamiento", "adopción",
                "paseo", "adiestramiento", "salud", "vacunas", "higiene", "correa", "pelaje",
                "jaula", "accesorios", "cachorro", "adulto", "juguetón", "amistad", "protección",
                "bienestar", "collar", "cama", "comportamiento", "snacks", "baño", "cepillo",
                "adiestramiento positivo", "alimentación balanceada", "pelota", "adiestrador"
            ],
            "gym": [
                "gimnasio", "entrenamiento", "fitness", "deporte", "actividad física", "pesas",
                "cardio", "rutina", "músculo", "resistencia", "bienestar", "motivación", "crossfit",
                "yoga", "pilates", "flexibilidad", "tonificación", "nutrición", "superación",
                "pesas libres", "pesas rusas", "cinta", "abdominales", "calentamiento", "enfriamiento",
                "deporte", "atleta", "rendimiento", "saludable", "descanso", "objetivos",
                "entrenador", "clases grupales", "entrenamiento funcional", "cardiovascular",
                "pesas olímpicas", "rutina personalizada", "ejercicio", "bienestar integral"
            ],
            "floristeria": [
                "florería", "flores", "arreglos florales", "decoración floral", "ramo", "bouquet",
                "rosas", "tulipanes", "margaritas", "naturaleza", "colores", "fragancia", "ornamentación",
                "evento", "regalo", "jardinería", "plantas", "follaje", "floral", "perfume",
                "orquídeas", "lirios", "claveles", "flores frescas", "flores secas", "arte floral",
                "decoración de eventos", "paisajismo", "plantas ornamentales", "flores naturales",
                "temporada", "exótico", "silvestre", "fresco", "boda", "celebración"
            ]
        }

        self.cargar_modelo()

    def guardar_puntuacion(self, slogan, rubro, modo, puntuacion):
        datos = []
        if os.path.exists(self.puntuaciones_file):
            with open(self.puntuaciones_file, "r", encoding="utf-8") as f:
                datos = json.load(f)
        registro = {
            "slogan": slogan,
            "rubro": rubro,
            "modo": modo,
            "puntuacion": puntuacion
        }
        datos.append(registro)
        with open(self.puntuaciones_file, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)

    def entrenar_modelo(self):
        if not os.path.exists(self.puntuaciones_file):
            print("No hay datos para entrenar.")
            return
        with open(self.puntuaciones_file, "r", encoding="utf-8") as f:
            datos = json.load(f)
        slogans = [d["slogan"] for d in datos]
        y = [d["puntuacion"] for d in datos]
        self.vectorizer = TfidfVectorizer(max_features=100)
        X = self.vectorizer.fit_transform(slogans)
        self.modelo = LinearRegression()
        self.modelo.fit(X, y)
        self.guardar_modelo()
        print("Modelo entrenado y guardado correctamente.")

    def guardar_modelo(self):
        with open(self.modelo_file, "wb") as f:
            pickle.dump(self.modelo, f)
        with open(self.vectorizer_file, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def cargar_modelo(self):
        if os.path.exists(self.modelo_file) and os.path.exists(self.vectorizer_file):
            with open(self.modelo_file, "rb") as f:
                self.modelo = pickle.load(f)
            with open(self.vectorizer_file, "rb") as f:
                self.vectorizer = pickle.load(f)

    def predecir_puntuacion(self, slogan):
        if self.modelo is None or self.vectorizer is None:
            return None
        X = self.vectorizer.transform([slogan])
        prediccion = self.modelo.predict(X)
        return prediccion[0]

    def elegir_mejor_slogan(self, candidatos, umbral_minimo=0):
        if self.modelo is None or self.vectorizer is None or not candidatos:
            return random.choice(candidatos) if candidatos else None

        puntuaciones = []
        for slogan in candidatos:
            X = self.vectorizer.transform([slogan])
            prediccion = self.modelo.predict(X)[0]
            puntuaciones.append((slogan, prediccion))

        filtrados = [s for s in puntuaciones if s[1] >= umbral_minimo]

        if not filtrados:
            return random.choice(candidatos)

        mejores_ordenados = sorted(filtrados, key=lambda x: x[1], reverse=True)
        return mejores_ordenados[0][0]

    def generar_slogan_popular(self, rubro):
        cache_file = f"cache_{rubro}.json"
        top_words = None

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    top_words = json.load(f)
            except Exception as e:
                print("Error leyendo cache:", e)

        if not top_words:
            try:
                pytrends = TrendReq(hl='es-AR', tz=180)
                kw_list = [rubro]
                pytrends.build_payload(kw_list, timeframe='today 3-m', geo='AR')

                data = pytrends.interest_over_time()
                if not data.empty:
                    top_words = [rubro]
                else:
                    related = pytrends.related_queries()
                    if related.get(rubro) and related[rubro].get('top') is not None:
                        top_words = related[rubro]['top']['query'].tolist()

            except Exception as e:
                print("Google Trends no respondió, usando datos locales...")

                fallback = {
                    "panaderia": ["pan", "masa", "hogar", "sabores", "croissant", "tradición", "horno"],
                    "petshop": ["mascotas", "cuidado", "amor", "compañía", "adopción", "juguetes", "veterinario"],
                    "gym": ["fuerza", "salud", "energía", "entrenamiento", "motivación", "resistencia", "disciplina"],
                    "floristeria": ["flores", "arreglos", "naturaleza", "colores", "bouquet", "fragancia", "decoración"],
                    "almacen": ["productos", "frescura", "vecindario", "confianza", "abarrotes", "mercado", "diario"],
                    "clases particulares": ["aprendizaje", "profesor", "alumno", "educación", "motivación", "éxito", "conocimiento"]
                }
                top_words = fallback.get(normalizar(rubro), ["tendencia", "popular", "actualidad"])

            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(top_words, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print("Error guardando cache:", e)

        plantillas_generales = [
            "{} que alegra tu día",
            "{} presente en cada detalle",
            "{} pensad@o@ para vos",
            "{} que transmite confianza",
            "{} que acompaña tu vida {}",
            "{} con dedicación y calidad",
            "{} que transforma tu mundo",
            "{} que inspira momentos únicos",
            "{} que hace la diferencia",
            "{} que conecta con tu esencia",
            "{} que despierta emociones",
            "{} que cuida lo que más querés",
            "{} que acompaña tu camino",
            "{} que renueva tu energía",
            "{} que refleja tu estilo",
            "{} que marca la diferencia",
            "{} que te acompaña siempre",
            "{} que impulsa tu pasión",
            "{} que transforma cada día",
            "{} que celebra la vida"
        ]

        plantillas_por_rubro = {
            "floristeria": [
                "{} que florece en tu día",
                "{} que perfuma tu mundo",
                "{} que ilumina cada ocasión",
                "{} que regala sonrisas",
                "{} que llena de color tu vida",
                "{} que embellece tus momentos",
                "{} que inspira con fragancia",
                "{} que decora tus sueños",
                "{} que hace vibrar la naturaleza",
                "{} que transforma espacios con flores",
                "{} que expresa tus sentimientos",
                "{} que crea atmósferas únicas"
            ],
            "panaderia": [
                "{} hornead@o@ con pasión",
                "{} que despierta tu mañana",
                "{} con tradición en cada bocado",
                "{} que endulza tus momentos",
                "{} que calienta tu hogar con sabor",
                "{} que cruje en cada mordida",
                "{} que lleva historia a tu mesa",
                "{} que nutre con amor",
                "{} que despierta tus sentidos",
                "{} que acompaña tus desayunos",
                "{} que hace especial cada día",
                "{} que une generaciones"
            ],
            "almacen": [
                "{} fresc@o@ cada día en tu barrio",
                "{} que conecta con tu vecindario",
                "{} que alimenta tu confianza",
                "{} siempre cerca de vos",
                "{} con la frescura de tu barrio",
                "{} que ofrece calidad y variedad",
                "{} que cuida tu economía",
                "{} que acompaña tu familia",
                "{} que hace tu compra fácil",
                "{} que mantiene viva la tradición",
                "{} que te brinda lo mejor",
                "{} que está en cada hogar"
            ],
            "petshop": [
                "{} para tu mejor amigo",
                "{} que cuida con amor",
                "{} pensad@o@ para tu mascota",
                "{} que acompaña cada ladrido y maullido",
                "{} que celebra la vida animal",
                "{} con cariño en cada detalle",
                "{} que protege su salud",
                "{} que hace feliz a tu peludo",
                "{} que entiende sus necesidades",
                "{} que acompaña su crecimiento",
                "{} que ofrece lo mejor para ellos",
                "{} que cuida su bienestar"
            ],
            "gym": [
                "{} que impulsa tu energía",
                "{} pensad@o@ para tu bienestar",
                "{} que fortalece tu rutina",
                "{} que motiva cada entrenamiento",
                "{} que te lleva a superar tus límites",
                "{} que transforma tu cuerpo y mente",
                "{} que acompaña tu esfuerzo",
                "{} que potencia tu rendimiento",
                "{} que te desafía a crecer",
                "{} que inspira tu disciplina",
                "{} que construye tu salud",
                "{} que renueva tu motivación"
            ],
            "clases particulares": [
                "{} que impulsa tu aprendizaje",
                "{} pensad@o@ para tu éxito",
                "{} que transforma tu estudio",
                "{} que guía tu conocimiento",
                "{} que motiva cada logro académico",
                "{} que abre puertas al futuro",
                "{} que potencia tus habilidades",
                "{} que acompaña tu crecimiento",
                "{} que hace fácil lo difícil",
                "{} que inspira tu curiosidad",
                "{} que construye tu confianza",
                "{} que prepara para el éxito"
            ]
        }

        generos_rubro = {
            "panaderia": "femenino",
            "petshop": "masculino",
            "gym": "masculino",
            "floristeria": "femenino",
            "almacen": "masculino",
            "clases particulares": "femenino"
        }

        rubro_norm = normalizar(rubro)
        genero = generos_rubro.get(rubro_norm, "masculino")

        plantillas = plantillas_por_rubro.get(rubro_norm, plantillas_generales)

        palabra = random.choice(top_words)
        slogan = random.choice(plantillas).format(palabra.capitalize(), rubro)

        if genero == "femenino":
            slogan = slogan.replace("@o@", "a")
        else:
            slogan = slogan.replace("@o@", "o")

        return slogan

    def obtener_sinonimos(self, rubro):
        rubro_norm = normalizar(rubro)
        sinonimos = set()
        for syn in wordnet.synsets(rubro_norm, lang="spa"):
            for lemma in syn.lemmas("spa"):
                sinonimos.add(lemma.name().replace('_', ' '))
        sinonimos = list(sinonimos)

        sinonimos_locales = self.diccionario_local.get(rubro_norm, [])
        sinonimos.extend(sinonimos_locales)
        sinonimos = list(set(sinonimos))

        if not sinonimos:
            sinonimos = ["creatividad", "innovación", "tradición", "originalidad", "imaginación", "ingenio", "pasión"]

        return sinonimos

    def generar_slogan_semantico(self, rubro):
        def normalizar_texto(texto):
            texto = texto.lower()
            texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
            return texto

        generos_rubro = {
            "panaderia": "femenino",
            "petshop": "masculino",
            "gym": "masculino",
            "floristeria": "femenino",
            "almacen": "masculino",
            "clases particulares": "femenino"
        }

        def adaptar_genero(plantilla, genero):
            if genero == "femenino":
                return plantilla.replace("{a}", "a").replace("{o}", "a")
            else:
                return plantilla.replace("{a}", "").replace("{o}", "o")

        rubro_norm = normalizar_texto(rubro)
        genero = generos_rubro.get(rubro_norm, "masculino")
        sinonimos = self.obtener_sinonimos(rubro_norm)

        doc = nlp(rubro_norm)
        palabras_spacy = [token.text for token in doc if not token.is_stop]

        try:
            similares = w2v_model.wv.most_similar(rubro_norm, topn=5)
            palabras_gensim = [w for w, _ in similares]
        except KeyError:
            palabras_gensim = ["inspiración", "energía", "pasión"]

        todas = list(set(sinonimos + palabras_spacy + palabras_gensim))

        palabras_excluir = ["popular", "tendencia", "emociones", "actualidad", "tiempo", rubro_norm]
        todas = [w for w in todas if len(w) > 3 and normalizar_texto(w) not in palabras_excluir]

        corpus_map = {
            "panaderia": corpus[0],
            "petshop": corpus[1],
            "gym": corpus[2],
            "floristeria": corpus[3],
            "almacen": corpus[4],
            "clases particulares": corpus[5]
        }
        palabras_rubro = corpus_map.get(rubro_norm, [])
        todas = [w for w in todas if w in palabras_rubro] or palabras_rubro

        if not todas:
            todas = ["creatividad", "innovación", "pasión"]

        palabra = random.choice(todas)

        def es_plural(palabra):
            palabra = palabra.lower()
            if palabra.endswith("es"):
                return True
            if palabra.endswith("s") and len(palabra) > 1 and palabra[-2] not in "aeiou":
                return True
            excepciones = ["crisis", "tesis", "parálisis", "análisis", "lunes", "tórax"]
            if palabra in excepciones:
                return False
            doc = nlp(palabra)
            for token in doc:
                return token.text.lower() != token.lemma_.lower()
            return False

        if es_plural(palabra):
            plantillas = [
                "{} que acompañan tu {}",
                "{} que inspiran tu {}",
                "{} que transforman tu {}",
                "{} presentes en cada {}"
            ]
        else:
            plantillas = [
                "{} que acompaña tu {}",
                "{} que inspira tu {}",
                "{} reflejado en tu {}",
                "{} como esencia de {}",
                "{} que transforma tu {}",
                "{} pensado para tu {}"
            ]
        plantilla_raw = random.choice(plantillas)
        plantilla = adaptar_genero(plantilla_raw, genero)
        return plantilla.format(palabra.capitalize(), rubro)

    def generar_slogan_diferenciado(self, rubro):
        return random.choice(self.diferenciados.get(rubro, ["Slogan diferenciado no disponible"]))

    def generar_slogan(self, rubro, modo="popular"):
        modo = normalizar(modo)
        if modo == "popular":
            return self.generar_slogan_popular(rubro)
        elif modo == "semantico":
            return self.generar_slogan_semantico(rubro)
        elif modo == "diferenciado":
            return self.generar_slogan_diferenciado(rubro)
        else:
            return "Modo no reconocido. Usa: popular, semántico o diferenciado."

    def generar_varios_slogans(self, rubro, modo, cantidad=3):
        candidatos = []
        for _ in range(cantidad):
            candidatos.append(self.generar_slogan(rubro, modo))
        return candidatos


def main():
    agente = AgenteSlogan()

    while True:
        print("\nRubros disponibles:", ", ".join(agente.diferenciados.keys()))
        print("Modos disponibles:", ", ".join(MODOS))

        rubro = input("¿Sobre qué rubro querés armar el slogan? ").strip()
        modo = input("Elegí modo: ").strip()

        rubro_norm = normalizar(rubro)
        modo_norm = normalizar(modo)

        rubro_map = {normalizar(k): k for k in agente.diferenciados.keys()}
        rubro_final = rubro_map.get(rubro_norm, rubro_norm)

        candidatos = agente.generar_varios_slogans(rubro_final, modo_norm, cantidad=3)
        mejor_slogan = agente.elegir_mejor_slogan(candidatos, umbral_minimo=2.5)
        print("\nMejor slogan elegido:", mejor_slogan)

        while True:
            try:
                puntuacion = int(input("Calificá el slogan del 1 (muy malo) al 5 (muy bueno): "))
                if 1 <= puntuacion <= 5:
                    break
                else:
                    print("Por favor, ingresá un número entre 1 y 5.")
            except ValueError:
                print("Entrada inválida. Ingresá un número entero.")

        agente.guardar_puntuacion(mejor_slogan, rubro_final, modo_norm, puntuacion)

        opcion = input("\n¿Querés generar otro slogan? (s/n): ").strip().lower()
        if opcion not in ["s", "si"]:
            print("Entrenando modelo con los datos recopilados, por favor espere...")
            agente.entrenar_modelo()
            print("¡Gracias por usar el generador de slogans! 👋")
            break


if __name__ == "__main__":
    main()
