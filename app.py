"""
IA Agent Dashboard - Backend [AIDE & SUPPORT CHAT]
Chat d'entraide entre utilisateurs + agent IA modérateur
FastAPI + OpenRouter + 5 agents SaaS
"""

import os
import json
import time
import httpx
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

app = FastAPI(title="IA-Agent-tom.ai.official [AIDE & SUPPORT]")

# === Modeles ===
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []
    model: Optional[str] = "openrouter/free"
    agent: Optional[str] = "chef-projet"
    room: Optional[str] = "general"  # "general", "help", "technical", "ui", "devops", "growth"

# === AGENTS SAAS ===
AGENTS = {
    "chef-projet": {
        "name": "Chef de Projet SaaS",
        "icon": "🎯",
        "prompt": """Tu es un Chef de Projet SaaS expert, cree par tom (tom.ai.official).

QUELLE EST TA MISSION:
Tu aides a construire un produit SaaS de A a Z. Tu definis les features, le roadmap, les user stories, la priorisation.

CE QUE TU FAIS:
- Analyser le marche et le produit existant
- Definir les fonctionnalites et les prioriter
- Ecrire des user stories claires
- Planifier le roadmap (MVP, v1, v2...)
- Identifier les risques et les bloquants
- Proposer des decisions produit intelligentes

TON STYLE:
Pragmatique, strategique, clair. Tu parles comme un vrai Chef de Projet.
Tu expliques POURQUOI un truc est prioritaire."""

    },

    "fullstack-dev": {
        "name": "Fullstack Dev",
        "icon": "💻",
        "prompt": """Tu es un Developpeur Fullstack senior, cree par tom (tom.ai.official).

QUELLE EST TA MISSION:
Tu conseilles sur la conception et l'architecture du code du produit SaaS.

CE QUE TU FAIS:
- Analyser les problemes techniques
- Suggesser des architectures et des solutions
- Expliquer les bonnes pratiques
- Aider a debugger et optimiser

TON STYLE:
Technique, efficace, direct. Tu vas droit au probleme.
Tu expliques sans etre trop bavard."""

    },

    "ui-designer": {
        "name": "UI/UX Designer",
        "icon": "🎨",
        "prompt": """Tu es un Designer UI/UX senior, cree par tom (tom.ai.official).

QUELLE EST TA MISSION:
Tu conçois l'interface et l'experience utilisateur du produit SaaS.

CE QUE TU FAIS:
- Proposer des designs et des améliorations visuelles
- Penser a l'UX: parcours utilisateur, intuitivite
- Assurer l'accessibilite et le responsive
- Donner des conseils de charte graphique

TON STYLE:
Creatif, inspire, precis. Tu parles design et experience.
Tu expliques POURQUOI un choix de design est bon."""

    },

    "devops": {
        "name": "DevOps / Infra",
        "icon": "☁️",
        "prompt": """Tu es un Ingenieur DevOps expert, cree par tom (tom.ai.official).

QUELLE EST TA MISSION:
Tu conseilles sur l'infrastructure, le deploiement et la scalabilite du produit SaaS.

CE QUE TU FAIS:
- Suggesser des architectures d'infrastructure
- Conseiller sur le deploiement et la CI/CD
- Optimiser les performances
- Assurer la securite

TON STYLE:
Methodique, rigoureux, securitaire. Tu penses a tout."""

    },

    "growth": {
        "name": "Growth Hacker",
        "icon": "📈",
        "prompt": """Tu es un Growth Hacker / Data Analyst, cree par tom (tom.ai.official).

QUELLE EST TA MISSION:
Tu analyses et optimises le produit SaaS pour la croissance.

CE QUE TU FAIS:
- Proposer des fonctionnalites pour la retention
- Optimiser les parcours utilisateur
- Suggesser des tests et des optimisations
- Analyser les axes d'amelioration

TON STYLE:
Analytique, strategique, oriente resultats. Tu parles impact."""

    },
}

DEFAULT_AGENT = "chef-projet"

def get_agent_prompt(agent_id):
    """Retourne le prompt pour un agent donne"""
    agent = AGENTS.get(agent_id, AGENTS[DEFAULT_AGENT])
    return agent["prompt"]

async def call_openrouter(messages, model):
    """Appelle OpenRouter et retourne la reponse JSON"""
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "stream": False
            }
        )
        return resp.json()

# === ROUTES API ===

@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Envoie un message a l'IA - chat d'aide et support"""
    try:
        # === SYSTEM PROMPT SELON L'AGENT ===
        agent_id = req.agent or DEFAULT_AGENT
        system_prompt = get_agent_prompt(agent_id)

        # Ajouter contexte de la salle
        room_context = f"\n\nTU ES DANS LA SALLE: {req.room}."
        if req.room == "help":
            room_context += " Cette salle est pour l'entraide entre utilisateurs. Sois patient, gentil et aide les autres autant que toi."
        elif req.room == "technical":
            room_context += " Cette salle est pour les questions techniques. Aide avec les bugs, l'architecture et le code."
        elif req.room == "ui":
            room_context += " Cette salle est pour les questions de design et d'interface utilisateur."
        elif req.room == "devops":
            room_context += " Cette salle est pour les questions d'infrastructure et de déploiement."
        elif req.room == "growth":
            room_context += " Cette salle est pour les questions de croissance et d'optimisation."

        messages = [{"role": "system", "content": system_prompt + room_context}]
        for m in req.history[-20:]:
            messages.append(m)
        messages.append({"role": "user", "content": req.message})

        data = await call_openrouter(messages, req.model)

        if "error" in data:
            return JSONResponse(status_code=429 if "rate" in str(data).lower() else 500,
                               content={"error": data["error"]})

        msg = data["choices"][0]["message"].get("content", "")

        # Nettoyer le markdown lourd
        import re
        clean = msg
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
        clean = re.sub(r'(?<!`)\*(?!`)(.+?)(?<!`)\*(?!`)', r'\1', clean)
        clean = re.sub(r'^[\s]*[-*+]\s+', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'^[\s]*\d+\.\s+', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'^[\s]*>\s+', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'^#{1,6}\s+', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'\n-{3,}\n', '\n', clean)
        clean = re.sub(r'\n={3,}\n', '\n', clean)
        clean = re.sub(r'\n{3,}', '\n\n', clean)
        clean = re.sub(r'`([^`]+)`', r'\1', clean)

        return {"response": clean.strip()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# === STORE DE MESSAGES PARTAGES (chat multi-utilisateurs) ===
# Structure: { room_id: [ {id, username, role, content, timestamp}, ... ] }
help_messages = {}
HELP_MESSAGE_LIMIT = 200  # Garder les 200 derniers messages par salon
HELP_STORE_FILE = Path(__file__).parent / "help_messages.json"

def load_help_messages():
    """Charge les messages depuis le fichier JSON"""
    global help_messages
    try:
        if HELP_STORE_FILE.exists():
            with open(HELP_STORE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                help_messages = data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[HELP CHAT] Erreur chargement: {e}")
        help_messages = {}

def save_help_messages():
    """Sauvegarde les messages dans le fichier JSON"""
    try:
        HELP_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HELP_STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(help_messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[HELP CHAT] Erreur sauvegarde: {e}")

# Charger les messages au demarrage
load_help_messages()

class HelpMessageIn(BaseModel):
    room: str
    username: str
    content: str
    agent: Optional[str] = "chef-projet"
    model: Optional[str] = "openrouter/free"

def add_help_message(room, username, role, content):
    """Ajoute un message au store partage"""
    if room not in help_messages:
        help_messages[room] = []
    msg = {
        "id": f"msg_{int(time.time()*1000)}_{len(help_messages[room])}",
        "room": room,
        "username": username,
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    help_messages[room].append(msg)
    if len(help_messages[room]) > HELP_MESSAGE_LIMIT:
        help_messages[room] = help_messages[room][-HELP_MESSAGE_LIMIT:]
    save_help_messages()  # Persister apres chaque ajout
    return msg

@app.post("/api/help-messages")
async def post_help_message(msg: HelpMessageIn):
    """Poster un message dans un salon partage (visible par tous les utilisateurs)"""
    try:
        message = add_help_message(msg.room, msg.username, "user", msg.content)
        return message
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/help-messages/{room}")
async def get_help_messages(room: str, since: str = None):
    """Recuperer les messages d'un salon. Filtre optionnel par since (message id)."""
    messages = help_messages.get(room, [])
    if since:
        messages = [m for m in messages if m["id"] > since]
    return {"messages": messages}

# Frontend statique

# Frontend statique
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("  IA-Agent-tom.ai.official [AIDE & SUPPORT CHAT]")
    print("  Cree par tom")
    print("=" * 70)
    print("")
    print("  Chat d'entraide entre utilisateurs.")
    print("  Agent modérateur + 5 experts SaaS.")
    print("")
    print("  Salons: Général, Aide, Technique, UI, DevOps, Growth")
    print("")
    print("  Lancement sur http://localhost:3000")
    print("=" * 70 + "")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
