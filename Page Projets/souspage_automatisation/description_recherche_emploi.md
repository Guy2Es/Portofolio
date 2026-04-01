# Workflow Automatisé : Recherche et Candidature aux Stages

Ce workflow automatise la recherche d'offres de stage, la génération de lettres de motivation personnalisées et l'enregistrement des candidatures dans un Google Sheet.

## Description générale

Le processus se déclenche automatiquement selon un planning (ou manuellement) et suit les étapes suivantes :

### 1. Déclenchement
- **Schedule Trigger** : Le workflow se lance automatiquement à intervalles réguliers (ex. : tous les jours).
- **Search URL (manuel)** : Possibilité de lancer le workflow manuellement en fournissant une URL de recherche d'offres (pour linkedin).

### 2. Récupération des offres de stage
- **Scrap Internship** : Scrape les offres de stage depuis l'url de recherche LinkedIn.
- **Scrap Internship1** : Second scraper (Indeed) pour récupérer davantage d’offres.

### 3. Fusion des données
- **Merge** : Combine les résultats des deux scrapers en une seule liste d’offres (supprime les doublons si besoin).

### 4. Analyse et filtrage avec l’IA
- **Verdict (Réponse Text)** : L’IA (type GPT) analyse chaque offre et donne un **verdict** boléen :
  - True = Bonne offre
  - False = pas d'interet 
  - basé sur mes critères de selection
- **Filter** : Ne conserve que les offres qui correspondent à mes critères (Réponse de gpt = True).

### 5. Génération de la candidature
- **Custom Cover Letter (Réponse Text)** : L’IA génère une **lettre de motivation personnalisée** pour chaque offre retenue, adaptée au poste et à l’entreprise.

### 6. Création des documents
- **Create a Document** : Crée un document Google Docs avec la lettre de motivation générée.
- **Add Letter txt** : Ajoute le texte de la lettre dans une colonne du Google Sheet (pour archivage rapide).

### 7. Enregistrement final
- **Append row in sheet** : Ajoute une ligne dans le Google Sheet avec toutes les informations :
  - Lien de l’offre
  - Nom de l’entreprise
  - Poste
  - nombre de candidats
  - Lien vers la lettre de motivation
  - Date de candidature d'apparition.
  - etc...

## Objectif du workflow

Automatiser et industrialiser ma recherche de stage en :
- Gagnant un temps considérable
- Personnalisant tes candidatures grâce à l’IA
- Gardant une traçabilité complète de toutes tes candidatures

---