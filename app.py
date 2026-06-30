"""
IA Agent Dashboard - Backend [AIDE & SUPPORT CHAT]
Chat d'entraide entre utilisateurs + agent IA modérateur
FastAPI + OpenRouter + 5 agents SaaS
"""

import os
import sys
import json
import time
import httpx
import subprocess
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Google Drive File ID pour la vérification des mises à jour
# 1. Upload version-template.json sur Google Drive
# 2. Partage -> "Tout le monde avec le lien"
# 3. Copie l'ID du fichier (ex: 1ABCxyz...)
# 4. Mets-le ici ou dans la variable d'env GOOGLE_DRIVE_UPDATE_ID
GOOGLE_DRIVE_UPDATE_ID = os.environ.get("GOOGLE_DRIVE_UPDATE_ID", "1Db6XxdPMVplJmxze7M7q7VPY4lN8xII6")
UPDATE_CHECK_URL = f"https://drive.google.com/uc?export=view&id={GOOGLE_DRIVE_UPDATE_ID}" if GOOGLE_DRIVE_UPDATE_ID else ""

# Version locale de l'application
APP_VERSION_DEFAULT = "3.0"
APP_VERSION_FILE = Path(__file__).parent / "current_version.txt"
APP_VERSION_URL = "https://github.com/tomtechclair/ia-agent-dashboard"

def load_app_version():
    """Charge la version depuis le fichier persistant, sinon utilise la valeur par défaut"""
    try:
        if APP_VERSION_FILE.exists():
            v = APP_VERSION_FILE.read_text(encoding="utf-8").strip()
            if v:
                print(f"[VERSION] Version chargée depuis le fichier : {v}")
                return v
    except Exception as e:
        print(f"[VERSION] Erreur lecture fichier version : {e}")
    print(f"[VERSION] Version par défaut : {APP_VERSION_DEFAULT}")
    return APP_VERSION_DEFAULT

def save_app_version(version):
    """Sauvegarde la version dans le fichier persistant"""
    try:
        APP_VERSION_FILE.write_text(version.strip(), encoding="utf-8")
        print(f"[VERSION] Version sauvegardée : {version}")
        return True
    except Exception as e:
        print(f"[VERSION] Erreur sauvegarde version : {e}")
        return False

APP_VERSION = load_app_version()

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

@app.get("/api/check-update")
async def check_update():
    """Verifie si une mise a jour est disponible via Google Drive ou fallback"""
    try:
        # Essayer de recuperer le fichier version depuis Google Drive
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(UPDATE_CHECK_URL)
            if resp.status_code == 200:
                # Google Drive renvoie parfois une page HTML au lieu du JSON
                content_type = resp.headers.get("content-type", "")
                if "text/html" in content_type.lower():
                    # Google Drive a affiche une page intermediaire
                    # On essaie avec confirm=t
                    resp2 = await client.get(UPDATE_CHECK_URL + "&confirm=t")
                    if resp2.status_code == 200:
                        remote = resp2.json()
                    else:
                        raise Exception(f"Google Drive blocked: HTTP {resp2.status_code}")
                else:
                    remote = resp.json()
            else:
                raise Exception(f"HTTP {resp.status_code}")
    except Exception as e:
        # Fallback : pas de reponse, on retourne simplement la version locale
        return {
            "current_version": APP_VERSION,
            "update_available": False,
            "error": str(e),
            "download_url": APP_VERSION_URL
        }

    remote_version = remote.get("latest_version", "0.0")
    remote_title = remote.get("title", "Nouvelle version disponible")
    remote_changes = remote.get("changes", [])
    remote_notes = remote.get("notes", "")
    # Toujours utiliser le lien GitHub comme URL de téléchargement
    # (le lien Google Drive est juste pour la vérification de version)
    remote_download = APP_VERSION_URL

    update_available = compare_versions(remote_version, APP_VERSION) > 0

    return {
        "current_version": APP_VERSION,
        "latest_version": remote_version,
        "update_available": update_available,
        "title": remote_title,
        "changes": remote_changes,
        "notes": remote_notes,
        "download_url": remote_download
    }

def compare_versions(v1, v2):
    """Compare deux versions 'X.Y'. Retourne >0 si v1 > v2, 0 si egal, <0 si v1 < v2"""
    try:
        parts1 = [int(x) for x in str(v1).split(".")]
        parts2 = [int(x) for x in str(v2).split(".")]
        for i in range(max(len(parts1), len(parts2))):
            p1 = parts1[i] if i < len(parts1) else 0
            p2 = parts2[i] if i < len(parts2) else 0
            if p1 != p2:
                return p1 - p2
        return 0
    except:
        return 0

@app.post("/api/apply-update")
async def apply_update():
    """Applique la mise a jour directement : git pull + restart"""
    try:
        repo_path = Path(__file__).parent
        output_lines = []
        new_version = None
        
        # 1. git fetch
        output_lines.append("🔍 Vérification des mises à jour...")
        fetch = subprocess.run(
            ["git", "fetch"],
            cwd=repo_path,
            capture_output=True, text=True, timeout=30
        )
        if fetch.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={"success": False, "output": [fetch.stderr.strip()]}
            )
        
        # 2. Voir les changements
        output_lines.append("📋 Récupération des changements...")
        log = subprocess.run(
            ["git", "log", "HEAD..origin/master", "--oneline", "--no-color"],
            cwd=repo_path,
            capture_output=True, text=True, timeout=15
        )
        if log.stdout.strip():
            commits = log.stdout.strip().split("\n")
            for c in commits[:10]:
                output_lines.append(f"  • {c}")
        
        # 3. git pull
        output_lines.append("⬇️  Téléchargement de la mise à jour...")
        pull = subprocess.run(
            ["git", "pull"],
            cwd=repo_path,
            capture_output=True, text=True, timeout=60
        )
        if pull.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={"success": False, "output": [pull.stderr.strip()]}
            )
        
        output_lines.append("✅ Mise à jour appliquée avec succès !")
        
        # 4. Déterminer la nouvelle version (3 sources de fiabilité décroissante)
        output_lines.append("💾 Enregistrement de la nouvelle version...")
        import re
        
        # Source A: Lire APP_VERSION_DEFAULT depuis le nouveau app.py (juste mis à jour par git pull)
        try:
            new_app_py = (repo_path / "app.py").read_text(encoding="utf-8")
            m = re.search(r'APP_VERSION_DEFAULT\s*=\s*["\']([^"\']+)["\']', new_app_py)
            if m:
                new_version = m.group(1).strip()
                output_lines.append(f"  → Version depuis le code: {new_version}")
        except Exception as e:
            output_lines.append(f"  ⚠️ Impossible de lire le nouveau code: {e}")
        
        # Source B: Google Drive (si la version du code a échoué)
        if not new_version:
            try:
                async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                    resp = await client.get(UPDATE_CHECK_URL)
                    if resp.status_code == 200:
                        content_type = resp.headers.get("content-type", "")
                        if "text/html" in content_type.lower():
                            resp2 = await client.get(UPDATE_CHECK_URL + "&confirm=t")
                            if resp2.status_code == 200:
                                remote = resp2.json()
                            else:
                                remote = None
                        else:
                            remote = resp.json()
                    else:
                        remote = None
                if remote:
                    gv = remote.get("latest_version", "").strip()
                    if gv:
                        new_version = gv
                        output_lines.append(f"  → Version depuis Google Drive: {new_version}")
            except Exception as e:
                output_lines.append(f"  ⚠️ Google Drive indisponible: {e}")
        
        # Source C: Fallback - garder la version actuelle
        if not new_version:
            new_version = APP_VERSION
            output_lines.append(f"  → Version actuelle conservée: {new_version}")
        
        # Toujours sauvegarder la version déterminée
        save_app_version(new_version)
        output_lines.append(f"  ✅ Version enregistrée: {new_version}")
        
        # 5. Redémarrer le serveur proprement
        output_lines.append("🔄 Redémarrage du serveur...")
        current_pid = os.getpid()
        
        # Utiliser le Python qui tourne actuellement (chemin ABSOLU)
        # Inutile de tester : c'est celui qui fait tourner le serveur, il marche forcément
        python_cmd = f'"{sys.executable}"'
        output_lines.append(f"  🔍 Python: {sys.executable}")
        
        restart_script = repo_path / "restart.bat"
        restart_script.write_text(
            f"@echo off\n"
            f"title IA-Agent Dashboard (redémarrage)\n"
            f"echo [1/4] Attente avant arrêt...\n"
            f"timeout /t 3 /nobreak >nul\n"
            f"echo [2/4] Arrêt de l'ancien serveur (PID {current_pid})...\n"
            f"taskkill /f /pid {current_pid} 2>nul\n"
            f"echo [3/4] Attente libération du port...\n"
            f"timeout /t 4 /nobreak >nul\n"
            f"echo [4/4] Démarrage du nouveau serveur...\n"
            f"cd /d \"{repo_path}\"\n"
            f"{python_cmd} app.py\n"
            f"pause\n"
        )
        
        # 🔥 CRITIQUE: Utiliser START pour créer une fenêtre INDÉPENDANTE
        # start crée un nouveau cmd.exe qui n'est PAS un enfant du processus Python
        # même si le parent est tué par taskkill, cette fenêtre continue
        subprocess.Popen(
            f'start "IA-Agent Update" cmd /c "{restart_script}"',
            shell=True,
            cwd=repo_path
        )
        
        # Retourner le resultat (le serveur sera tué 3s plus tard par restart.bat)
        return {
            "success": True,
            "output": output_lines,
            "restarting": True
        }
        
    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=500,
            content={"success": False, "output": ["⏱️ La mise à jour a pris trop de temps."]}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "output": [f"❌ Erreur: {str(e)}"]}
        )

@app.get("/api/version")
async def get_version():
    """Retourne la version actuelle de l'application"""
    return {
        "version": APP_VERSION,
        "version_url": APP_VERSION_URL
    }

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
