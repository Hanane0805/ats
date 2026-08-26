"""
rag/chunking.py
===============
Découpage du CV en chunks exploitables pour le pipeline RAG.
"""

import re

# Pattern de détection des sections classiques d'un CV
SECTION_PATTERN = r'(?=(?:^|\n)\s*(?:'
SECTION_PATTERN += r'experience|expérience|parcours|emploi|career|work|job|poste|'
SECTION_PATTERN += r'formation|education|diplome|diploma|degree|university|université|bachelor|master|phd|'
SECTION_PATTERN += r'baccalaureat|baccalauréat|'
SECTION_PATTERN += r'competences?|skills?|technologies?|outils?|'
SECTION_PATTERN += r'langues?|languages?|projets?|projects?|certifications?|'
SECTION_PATTERN += r'interets?|interests?|profil|about|objective|resume|summary|references?)\s*[:.\n])'


def chunk_cv(cv_text, chunk_size=500, overlap=50):
    """Découpe un CV texte brut en chunks exploitables."""
    sections = re.split(SECTION_PATTERN, cv_text, flags=re.I)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section or len(section) < 30:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            start = 0
            while start < len(section):
                end = start + chunk_size
                chunk = section[start:end]
                if len(chunk.strip()) >= 30:
                    chunks.append(chunk.strip())
                start += chunk_size - overlap

    if not chunks:
        words = cv_text.split()
        current_chunk = []
        current_len = 0
        overlap_words = overlap // 3
        for word in words:
            current_chunk.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap_words:]
                current_len = sum(len(w) + 1 for w in current_chunk)
        if current_chunk:
            chunks.append(" ".join(current_chunk))

    return chunks