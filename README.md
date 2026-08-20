# 🤖 ATS - Système Intelligent de Recrutement

Système ATS intelligent propulsé par un modèle BERT pour l'analyse et le scoring automatique des candidatures.

## 📁 Architecture du Projet

| Composant | Tech | Lien | Port |
|-----------|------|------|------|
| **Backend** | Spring Boot (Java) | [ats-intelligent-backend](https://github.com/Hanane0805/ats-intelligent-backend) | 8080 |
| **Frontend** | Next.js 16 | Dépôt frontend | 3000 |
| **AI Service** | Flask + BERT | Ce dépôt | 5000 |

## 🚀 Installation

### Prérequis
- Python 3.9+
- pip

### Étapes

**1. Cloner le dépôt**

git clone https://github.com/Hanane0805/ats.git
cd ats

2. Créer l'environnement virtuel

python -m venv venv

Windows : venv\Scripts\activate
Linux/Mac : source venv/bin/activate
3. Installer les dépendances
pip install -r requirements.txt
4. Télécharger le modèle BERT

Le modèle (678 Mo) n'est pas inclus dans le dépôt. Téléchargez-le depuis :
(https://drive.google.com/drive/folders/1NFole_GbIvp43pbAQC-44BJ0K5fmKuI6?usp=sharing)

Structure attendue :
ats/
├── app.py
├── requirements.txt
├── ats_search.json
├── .gitignore
├── README.md
└── models/
    └── ats_bert_model/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── training_args.bin
5. Démarrer le service

python app.py
Le service démarre sur http://localhost:5000
⚙️ Configuration
Le service Flask tourne sur le port 5000. Le backend Spring Boot communique via :

properties

flask.ai.service.url=http://localhost:5000
🔄 Workflow
Un candidat poste sa candidature avec un CV (PDF)
Le backend extrait le texte du CV avec Apache Tika
Le backend envoie CV + offre au service Flask
Le modèle BERT analyse et calcule les scores
Flask renvoie les scores + recommandations en français
Le recruteur voit les résultats sur le frontend
🛠 Technologies
Python 3.9+
Flask
Transformers (HuggingFace)
PyTorch
scikit-learn
Apache Tika (côté backend)
