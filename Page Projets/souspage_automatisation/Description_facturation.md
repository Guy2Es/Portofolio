# Workflow Facturation Automatique via Telegram

Ce workflow permet de creer et envoyer des factures automatiquement des qu'un utilisateur envoie un message, une photo, un audio ou un PDF sur un bot Telegram.

## Objectif
Transformer n'importe quel envoi (texte, image, vocal ou fichier) en une facture PDF professionnelle envoyee automatiquement au client.

---

## Fonctionnement General

1. Declenchement : Un message arrive sur le bot Telegram
2. Switch intelligent : Analyse le type de contenu et dirige vers la bonne branche
3. Traitement selon le type (texte, image, audio, PDF)
4. AI Agent : Analyse, extrait les informations et calcule la facture
5. Generation du PDF
6. Envoi de la facture au client sur Telegram

---

## Les Branches d'Entree

### A. Branche Audio (Message vocal)
- Get a audio : Recupere le fichier audio
- Transcribe a recording (Whisper) : Convertit la voix en texte
- Resultat : Envoye a l'AI Agent

### B. Branche Image (Photo de ticket, facture, etc.)
- Get a image : Recupere l'image
- Mise au bon format : Prepare l'image
- Analyze image (Vision OpenAI) : Extrait le texte et decrit l'image
- Resultat : Envoye a l'AI Agent

### C. Branche Fichier (PDF ou autres fichiers)
- Get a file : Recupere le fichier
- Envoie PDF : Envoie vers l'API OCR
- Reception URL : Recupere le lien du resultat
- Resultat OCR : Extrait tout le texte du document
- Resultat : Envoye a l'AI Agent

### D. Branche Texte simple
- Le texte est envoye directement a l'AI Agent

---

## Le Coeur du Systeme : AI Agent

L'AI Agent est le cerveau du workflow. Il recoit toutes les informations et dispose de :

- Modele : OpenAI Chat Model (GPT-4o ou equivalent)
- Memoire : Simple Memory (contexte des conversations precedentes)
- Outils integres :
  - Calculator : Pour tous les calculs (TVA, totaux, remises...)
  - Informations de facturation : Lit une feuille (Google Sheet) contenant clients, tarifs, etc.

Role de l'Agent :
- Comprendre la demande
- Extraire les produits/services
- Identifier le client
- Calculer les montants
- Structurer les donnees en JSON

---

## Sortie et Generation de la Facture

1. Transformation en objet JSON : Donnees propres et structurees
2. Create a pdf : Genere le PDF de facture
3. Basic LLM Chain : (Optionnel) Reformule un message d'accompagnement
4. Send a text message : Envoie le PDF + message sur Telegram