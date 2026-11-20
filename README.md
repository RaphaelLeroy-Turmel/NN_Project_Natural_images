# Neural Network CNN : classification d'images

Projet d'équipe classification d'images avec un CNN sur le dataset **Natural Images** (Kaggle, `prasunroy/natural-images`).

- 8 classes : airplane, car, cat, dog, flower, fruit, motorbike, person
- 6 899 images, répartition plutot equilibrée (dog minoritaire avec 10.2% et et furit majoritaire avec 14.5% , moyenne des 10 classe)

## Attendus du projet

- Un modèle de classification basé sur un CNN
- Au moins une approche de transfer learning
- Des expériences ou améliorations supplémentaires
- Présentation finale du projet, puis questions orales individuelles sur le code, les choix d'implémentation et les concepts

## Répartition du travail

| Membre | Piste | Contenu |
|---|---|---|
| Pierre-Louis | **A** : CNN from scratch | Baseline construit à la main, avec et sans data augmentation |
| Florentin | **B** : Transfer learning (extraction de features) | Backbone pré-entraîné gelé + nouvelle tête de classification |
| Raphaël | **C** : Fine-tuning | Dégel des dernières couches d'un backbone, éventuellement un second modèle |

## Étapes

### 1. Socle commun (ensemble)

- Installer l'environnement et télécharger le dataset avec `kagglehub`
- Explorer le dataset (nombre de classes, tailles, répartition)
- Créer **un split unique** train / validation / test, stratifié, avec une seed fixe, sauvegardé dans un fichier CSV partagé
- Fixer une **taille d'image commune** (ex. 224×224)
- Fixer les **métriques communes** : accuracy, F1 macro, matrice de confusion (calculées sur le même jeu de test)
- Écrire un module de chargement des données partagé, utilisé par tous
- Choisir un format de résultats commun (un fichier JSON ou CSV par expérience : modèle, hyperparamètres, métriques)

### 2. Pistes en parallèle (chacun sa piste)

- Entraîner le modèle de sa piste en utilisant le split et les métriques communes
- Tester des variantes (learning rate, optimiseur, dropout, augmentation...)
- Sauvegarder les résultats au format commun

### 3. Analyse commune

- Tableau comparatif des modèles (métriques, temps d'entraînement, nombre de paramètres)
- Comparaison des courbes d'apprentissage
- Analyse des erreurs : quelles classes sont confondues et pourquoi
- Expériences supplémentaires (ex. ablation de la data augmentation, visualisation de type Grad-CAM)

### 4. Préparation de la présentation finale

- Rédiger la présentation et les résultats principaux
- Séance de préparation à l'oral : chacun explique son code aux deux autres
- S'entraîner à répondre aux questions « pourquoi ce choix ? » sur **tout** le projet, pas seulement sur sa piste

## Règles d'équipe

- Tout le monde utilise le même split, la même taille d'image et les mêmes métriques
- Un dépôt Git partagé, une branche par piste (format du nom de branche : modele A : from scratch)
- Chaque membre doit comprendre et pouvoir expliquer l'ensemble du projet