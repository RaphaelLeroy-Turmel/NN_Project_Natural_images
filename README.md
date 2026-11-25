# Neural Network CNN : classification d'images

Projet d'équipe (3 étudiants, ESILV) : classification d'images avec un CNN sur le dataset **Natural Images** (Kaggle, `prasunroy/natural-images`).

- 8 classes : airplane, car, cat, dog, flower, fruit, motorbike, person
- 6 899 images, répartition légèrement déséquilibrée (de 702 à 1 000 images par classe)

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

## Structure du projet

```text
main.py
data/
    train/
    valid/
    test/
src/
    model1.py
    model2.py
    ...
results/
    model1_v1/
        weights
        train_loss.csv
        valid_loss.csv
        test_output.json    => {"metric1": val, ...}
        config.yaml (correspond au configuration du modele : nombre de parametre, activation function , etc...)
    model1_v2/
        ...
```

## Étapes

### 1. Socle commun (ensemble)

- Installer l'environnement et télécharger le dataset avec `kagglehub`
- Explorer le dataset (nombre de classes, tailles, répartition)
- Créer **un split unique** train / validation / test, stratifié, avec une seed fixe, sauvegardé dans un fichier CSV partagé
- Fixer une **taille d'image commune** (128 × 128 pixels, RGB)
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
- Un dépôt Git partagé, une branche par piste
- Chaque membre doit comprendre et pouvoir expliquer l'ensemble du projet
## Vérifier le socle commun

Depuis la racine du dépôt, installez les dépendances avec
`python -m pip install -r requirements.txt`. Préparez les dossiers
`data/train`, `data/valid` et `data/test` avec la cellule de copie du
notebook et le fichier partagé `splits.csv`.

Lancez `python main.py` pour vérifier les effectifs, les fichiers et les
deux méthodes de redimensionnement sur la validation. Le redimensionnement
direct sert de référence. L'option avec marges conserve les proportions.
Les deux méthodes utilisent les mêmes chemins et le même ordre de classes.
Les pixels restent entre 0 et 255 au chargement.

Pour sauvegarder une expérience, utilisez `save_results` dans
`src/evaluation.py`. La matrice de confusion utilise les classes du
chargeur dans l'ordre alphabétique, avec les vraies classes en lignes et
les prédictions en colonnes.

### Variantes et robustesse du modèle B

Depuis la racine du dépôt, avec TensorFlow et Matplotlib disponibles :

```bash
python -m src.model_b --resize direct
python -m src.model_b --resize padding
python -m src.robustness_b --resize direct
python -m src.robustness_b --resize padding
```

Chaque entraînement écrit ses poids, sa configuration, ses courbes et ses
scores dans `results/model_b_direct_v1/` ou `results/model_b_padding_v1/`.
Relancer une variante remplace ses résultats dans le dossier correspondant.

La robustesse réutilise les poids de la variante choisie et évalue les mêmes
images de test à plusieurs niveaux de flou, de pixelisation et d'occlusion.
Le résultat est enregistré dans `results/model_b_<resize>_robustness_v1/`
avec les courbes, le CSV des scores, le JSON et un fichier de configuration
qui indique les poids utilisés. Pour la pixelisation, un bloc de 1 px
correspond au test original, et un bloc de 2 px à la réduction en 64 × 64
suivie du retour à 128 × 128.
