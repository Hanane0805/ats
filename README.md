# 🤖 ATS - Système Intelligent de Recrutement

Système ATS intelligent propulsé par un modèle BERT pour l'analyse et le scoring automatique des candidatures, enrichi d'un pipeline **RAG** (Retrieval-Augmented Generation) avec pgvector et LLM local (Ollama).

---

## 📁 Architecture du Projet

| Composant | Tech | Port | Rôle |
|-----------|------|------|------|
| **Backend** | Spring Boot (Java) | 8080 | API métier, extraction PDF (Apache Tika) |
| **Frontend** | Next.js 16 | 3000 | Interface recruteur |
| **AI Service** | Flask + BERT + RAG | **5000** | Analyse, scoring, génération LLM |

---

## 🗂️ Structure du AI Service (ce dépôt)

```
ats/
├── app.py                  # Point d'entrée Flask (Port 5000)
├── config.py               # Configuration DB, Ollama, logging
├── requirements.txt        # Dépendances Python
├── .env.example            # Variables d'environnement
│
├── core/                   # Composants fondamentaux
│   ├── models.py           # Chargement lazy BERT + SentenceTransformer
│   └── database.py         # Connexion PostgreSQL + init pgvector
│
├── rag/                    # Pipeline Retrieval-Augmented Generation
│   ├── chunking.py         # Découpage du CV en chunks
│   ├── storage.py          # Stockage chunks + embeddings dans pgvector
│   ├── retrieval.py        # Recherche sémantique par similarité cosinus
│   └── llm.py              # Appel Ollama (llama3.2:3b)
│
├── scoring/                # Algorithmes de scoring
│   ├── heuristic.py        # 14 règles ATS (130 pts)
│   ├── compatibility.py    # Score CV-Offre (4 dimensions pondérées)
│   └── bert_scorer.py      # Classification BERT (classe 0/1)
│
├── extraction/             # Extraction d'informations
│   └── info_extractor.py   # Email, téléphone, nom, skills, expérience
│
└── api/                    # Endpoints Flask (Blueprints)
    ├── routes.py           # Registre des blueprints
    ├── health.py           # /health, /rag/health
    ├── analyze.py          # /rag/analyze, /rag/analyze_rag, /rag/index_cv
    ├── embed.py            # /rag/embed
    ├── search.py           # /rag/search
    ├── extract.py          # /rag/extract
    └── compatibility.py    # /rag/compatibility
```

---

## 🚀 Installation

### Prérequis
- Python 3.9+
- PostgreSQL avec l'extension **pgvector**
- Ollama (optionnel, pour l'analyse LLM)

### 1. Cloner le dépôt
```bash
git clone https://github.com/Hanane0805/ats.git
cd ats
```

### 2. Environnement virtuel
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Télécharger le modèle BERT

Le modèle (678 Mo) n'est pas inclus dans le dépôt. Téléchargez-le depuis :
👉 [Google Drive](https://drive.google.com/drive/folders/1NFole_GbIvp43pbAQC-44BJ0K5fmKuI6?usp=sharing)

**Structure attendue :**
```
ats/
├── app.py
├── requirements.txt
├── ...
└── models/
    └── ats_bert_model/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── training_args.bin
```

### 5. Configurer la base de données

Copiez le fichier d'environnement :
```bash
cp .env.example .env
```

Modifiez `.env` selon votre configuration PostgreSQL :
```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=ats_db
DB_USER=postgres
DB_PASSWORD=postgres
OLLAMA_URL=http://localhost:11434
```

**Note :** L'extension `pgvector` doit être installée sur PostgreSQL :
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 6. Démarrer le service
```bash
python app.py
```

Le service démarre sur **http://localhost:5000**

---

## ⚙️ Configuration Spring Boot

Dans le backend Spring Boot, configurez l'URL du service AI :

```properties
flask.ai.service.url=http://localhost:5000
```

---

## 🔄 Workflow Complet

```
1. Candidat poste sa candidature avec un CV (PDF)
   │
   ▼
2. Backend Spring Boot extrait le texte du CV avec Apache Tika
   │
   ▼
3. Backend envoie CV + offre d'emploi au service Flask (Port 5000)
   │
   ▼
4. AI Service analyse :
   ├─ Score heuristique ATS (14 règles)
   ├─ Classification BERT (modèle fine-tuné)
   ├─ Chunking + Retrieval RAG (pgvector)
   └─ Analyse LLM (llama3.2:3b via Ollama)
   │
   ▼
5. Flask renvoie les scores + recommandations en français
   │
   ▼
6. Le recruteur voit les résultats sur le frontend Next.js
```

---

## 🧮 Les Scores

### Score Heuristique ATS (130 pts)
14 règles regex vérifiant la présence d'éléments essentiels :
- Email, téléphone, sections (expérience, formation, compétences)
- Dates, mots-clés techniques, titres de poste
- Réalisations quantifiées, acronymes expliqués, ordre chronologique

### Score BERT
Classification neuronale du CV (classe 0 = non ATS-friendly, 1 = ATS-friendly).

### Score de Compatibilité CV ↔ Offre
4 dimensions pondérées :
| Dimension | Poids | Description |
|-----------|-------|-------------|
| Similarité sémantique | 40% | Cosinus entre embeddings CV et offre |
| Match compétences | 30% | Intersection skills techniques |
| Match expérience | 20% | Comparaison années d'expérience |
| Mots-clés (Jaccard) | 10% | Correspondance vocabulaire |

### Score Hybride Combiné (Pipeline RAG)
Fusion pondérée des 3 sources :
| Source | Poids | Condition |
|--------|-------|-----------|
| BERT | 40% | Si le modèle est chargé |
| Heuristique | 30% | Toujours disponible |
| LLM | 30% | Si Ollama répond avec un score |

---

## 🔄 Pipeline RAG

```
CV brut (texte extrait par Tika)
   │
   ▼
┌─────────────┐
│  Chunking   │  ← Découpe par sections + fenêtres glissantes
└─────────────┘
   │
   ▼
┌─────────────┐
│  Embedding  │  ← MiniLM-L12-v2 (384d)
└─────────────┘
   │
   ▼
┌─────────────┐
│  pgvector   │  ← Stockage vectoriel PostgreSQL
└─────────────┘
   │
   │  Offre d'emploi (embedding de la requête)
   ▼
┌─────────────┐
│  Retrieval  │  ← Top-k chunks par similarité cosinus
└─────────────┘
   │
   ▼
┌─────────────┐
│    LLM      │  ← llama3.2:3b génère l'analyse
│  (Ollama)   │    à partir des chunks + contexte
└─────────────┘
   │
   ▼
Analyse JSON (points forts, faiblesses, compatibilité, recommandations)
```

---

## 🛠 Technologies

- **Python 3.9+**
- **Flask** + Flask-CORS
- **Transformers** (HuggingFace) + PyTorch
- **Sentence-Transformers** (paraphrase-multilingual-MiniLM-L12-v2)
- **PostgreSQL** + **pgvector**
- **psycopg2-binary**
- **Ollama** (llama3.2:3b)
- **Apache Tika** (côté backend Spring Boot)

---

## 🧪 Tests Unitaires

Chaque module peut être testé indépendamment sans charger Flask :

```python
# Test du chunking
from rag.chunking import chunk_cv
chunks = chunk_cv("Mon CV avec expérience...")

# Test du scoring heuristique
from scoring.heuristic import ats_score
result = ats_score("Mon CV...")

# Test de compatibilité
from scoring.compatibility import compute_compatibility_score
result = compute_compatibility_score(cv_text, job_desc)
```

---

## 📡 Endpoints API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Santé du service |
| `/api/rag/health` | GET | Santé de tous les composants |
| `/api/rag/analyze` | POST | Analyse ATS (BERT + heuristique) |
| `/api/rag/analyze_rag` | POST | Pipeline RAG complet |
| `/api/rag/index_cv` | POST | Indexation d'un CV dans pgvector |
| `/api/rag/embed` | POST | Stockage embedding global |
| `/api/rag/search` | POST | Recherche sémantique |
| `/api/rag/extract` | POST | Extraction infos structurées |
| `/api/rag/compatibility` | POST | Score CV ↔ Offre |

---

## 👤 Auteur

[@Hanane0805](https://github.com/Hanane0805)
