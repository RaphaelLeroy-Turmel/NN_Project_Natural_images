# Finalistes — piste C (fine-tuning, MobileNetV2)

Choisis sur la validation (loss moyenne sur les seeds 42, 43, 44), avant toute lecture du test.
Budget de réglage : 8 configurations essayées (grille), 3 présélectionnées.

| Catégorie | Config | val_loss (moy ± std) | val_f1_macro (moy) |
|---|---|---|---|
| Backbone gelé | unfreeze=0, lr_head=1e-3 | 0,0124 ± 0,0020 | 0,9968 |
| Fine-tuning | unfreeze=30, lr_finetune=1e-4 | 0,0108 ± 0,0017 | 0,9970 |

Écarté : unfreeze=10, lr_finetune=1e-4 (0,0151 ± 0,0029), dont l'intervalle ne recoupe pas celui de 30 couches.