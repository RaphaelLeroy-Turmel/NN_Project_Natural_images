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
main.py                       # vérification du socle commun (ne pas y mettre de piste)
run_finetuning.py             # piste C : charge les données et lance les expériences
requirements.txt
splits.csv                    # split train / validation / test partagé
Data_Exploration.ipynb
data/
    train/
    valid/
    test/
src/
    data_load.py             # chargement des données partagé (load_images, IMAGE_SIZE)
    evaluation.py            # métriques communes et save_results
    degradation.py           # pollution des images d'entraînement (partagé)
    model_finetuning.py      # piste C : MobileNetV2, tête de classification, gel / dégel
    experiment.py            # piste C : run_experiment (entraîner, évaluer, sauvegarder)
    model1.py                # autres pistes : un fichier par modèle
    ...
results/
    <nom_du_modèle>_v<n>/    # un dossier par expérience, jamais écrasé
        weights.weights.h5
        train_loss.csv
        valid_loss.csv
        test_output.json    => {"accuracy": val, "f1_macro": val, "confusion_matrix": [...], ...}
        config.yaml (correspond au configuration du modele : nombre de parametre, activation function , etc...)
```

Convention de nommage des dossiers de `results/` (piste C) :
`C_<backbone>_<phase>_lr<lr>_<pollution>_v<n>`, par exemple
`C_mobilenetv2_unfreeze30layers_lr1e-05_flou20_v1` (30 couches dégelées,
learning rate 1e-5, 20 % des images d'entraînement floutées). Le taux de
pollution est aussi enregistré dans `config.yaml`.

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
