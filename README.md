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
run_model_b.py
plot_comparison.py
compare_pollution.py
data/
    train/
    valid/
    test/
src/
    data_load.py             # chargement des données partagé (load_images, IMAGE_SIZE)
    evaluation.py            # métriques communes et save_results
    degradation.py           # pollution des images d'entraînement (partagé)
    model1.py                # piste A : CNN from scratch (4 blocs convolutifs, augmentation optionnelle)
    experiment_a.py          # piste A : train_run, evaluate_on_test, finalistes
    model_b.py
    robustness_b.py
    model_finetuning.py      # piste C : MobileNetV2, tête de classification, gel / dégel
    experiment.py            # piste C : run_experiment (entraîner, évaluer, sauvegarder)
    runner.py
    runs.py
    ...
track_A_pierre_louis/
    exploration_and_results.ipynb   # piste A : réglage, finalistes, test, pollution, robustesse
results/
    <nom_du_run>/            # un dossier par run, jamais écrasé
        model.weights.h5     # non versionné dans Git
        train_loss.csv
        valid_loss.csv
        test_output.json    => {"accuracy": val, "f1_macro": val, "confusion_matrix": [...], ...}
        config.yaml (correspond au configuration du modele : nombre de parametre, activation function , etc...)
    finalistes_<piste>.md    # finalistes désignés par chaque piste (voir le protocole)
```

Convention de nommage des dossiers de `results/` : le nom contient la piste, le modèle, la
configuration, le niveau de pollution (`clean` si aucun) et la seed. Exemple :
`C_mobilenetv2_unfreeze30layers_lr1e-05_flou20_seed42` (piste C, 30 couches dégelées, learning
rate 1e-5, 20 % des images d'entraînement floutées, seed 42). Un run déjà terminé n'est pas
relancé. Le taux de pollution est aussi enregistré dans `config.yaml`.

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

- Régler les modèles de sa piste sur la validation (le test n'est pas évalué), avec le split et les métriques communes
- Tester des variantes (learning rate, optimiseur, dropout, augmentation...) et noter combien de configurations ont été essayées
- Désigner un finaliste par catégorie de modèle, puis le mesurer sur le test avec plusieurs seeds (voir le protocole expérimental commun)
- Sauvegarder les résultats au format commun

### 3. Analyse commune

- Tableau comparatif des finalistes (métriques sur le test, temps d'entraînement, nombre de paramètres)
- Étude de pollution : les finalistes rejoués à des niveaux de pollution croissants, courbes des trois pistes sur les mêmes axes
- Comparaison des courbes d'apprentissage
- Analyse des erreurs : quelles classes sont confondues et pourquoi
- Expériences supplémentaires (ex. ablation de la data augmentation, visualisation de type Grad-CAM)

### 4. Préparation de la présentation finale

- Rédiger la présentation et les résultats principaux
- Séance de préparation à l'oral : chacun explique son code aux deux autres
- S'entraîner à répondre aux questions « pourquoi ce choix ? » sur **tout** le projet, pas seulement sur sa piste

## Protocole expérimental commun

Ce protocole s'applique aux trois pistes, pour que leurs résultats soient comparables. Les valeurs marquées *(proposition)* sont à valider ensemble.

| Règle | Pourquoi |
|---|---|
| Un seul split (`splits.csv`) pour tous les modèles et toutes les pistes | Comparaison équitable, et image par image |
| La validation sert à choisir, le test sert à mesurer. Le test n'est lu que pour les finalistes et pour l'étude de pollution | Si l'on choisit d'après le test, son score devient optimiste |
| Critère principal : loss de validation. Contrôles : F1 macro et matrice de confusion. L'accuracy est rapportée mais ne tranche pas | Avec environ 1 000 images de validation, une image vaut 0,1 point et l'accuracy est déjà à 99 % |
| Arrêt anticipé identique : loss de validation, patience 3, restauration des meilleurs poids | Même règle de fin d'entraînement pour tous |
| Si deux variantes sont dans le bruit, on garde la plus simple ou la moins coûteuse | On ne paie pas une complexité pour un gain indistinguable du hasard |
| 3 seeds (42, 43, 44, *proposition*), moyenne ± écart-type, jamais la meilleure seed | Mesure la variabilité due à l'entraînement |
| Budget de réglage comparable : chaque piste note le nombre de configurations essayées | Plus on essaie de variantes sur la validation, plus on la sur-ajuste |
| Pollution : `pollute_train` (`src/degradation.py`), train uniquement, pixels bruts avant normalisation, mêmes niveaux et mêmes seeds de pollution pour tous (0, 10, 20 et 30 % avec flou, *proposition*) | Courbes superposables entre les pistes |
| Pas de ré-optimisation des hyperparamètres par niveau de pollution | Chaque courbe montre uniquement l'effet de la pollution |

Déroulé : 1. régler sur la validation, 2. présélectionner, 3. choisir un finaliste par catégorie avec plusieurs seeds, 4. mesurer les finalistes sur le test, 5. étude de pollution, 6. analyse commune. Les finalistes sont désignés et partagés avant de lire un seul score de test.

## Règles d'équipe

- Tout le monde utilise le même split, la même taille d'image, les mêmes métriques et le protocole expérimental commun
- Un dépôt Git partagé, une branche par piste
- Les fichiers partagés (`src/degradation.py`, `src/data_load.py`, `src/evaluation.py`, `README.md`, `requirements.txt`) ne se modifient qu'après avoir prévenu l'équipe
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
## Piste A : CNN from scratch

Le notebook `track_A_pierre_louis/exploration_and_results.ipynb` suit le protocole commun, partie
par partie : réglage sur la validation (seed 42), finalistes sur 3 seeds, mesure sur le test,
étude de pollution, puis un bonus de robustesse au test au même format que la piste B. Il se lance
depuis Jupyter, et les runs déjà présents dans `results/` (fichier `config.yaml`) ne sont pas
réentraînés.

Modèle : 4 blocs `Conv2D-BatchNorm-ReLU-MaxPool` (32, 64, 128, 128 filtres), global average
pooling, `Dense(128)`, dropout 0,3, softmax. 259 784 paramètres. Les pixels sont ramenés entre 0
et 1 dans le modèle (`Rescaling(1/255)`). Deux catégories : sans et avec data augmentation
(flip horizontal, rotation, zoom, contraste).

Finalistes (`results/finalistes_A.md`) : `A_cnn4blocks_noaug_lr0.0003_do0.3` et
`A_cnn4blocks_aug_lr0.0003_do0.3`, 2 configurations essayées par catégorie.

| Finaliste | Accuracy test | F1 macro test |
|---|---|---|
| sans augmentation | 83,2 % ± 7,4 | 0,814 ± 0,076 |
| avec augmentation | 82,3 % ± 3,0 | 0,807 ± 0,028 |

Moyenne ± écart-type sur les seeds 42, 43, 44. Les autres résultats sont dans `results/` :
`test_A.csv`, `confusion_A.png`, `pollution_A.csv`, `pollution_A.png` et les dossiers
`*_robustness/`.


## Piste C : fine-tuning