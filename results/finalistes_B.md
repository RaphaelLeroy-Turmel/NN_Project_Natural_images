# Finaliste de la piste B

Redimensionnement retenu : `padding`

Configurations essayées : 2. Seeds : 42, 43, 44.

Loss de validation (moyenne ± écart-type) :
- direct : 0.012059 ± 0.001770
- padding : 0.010145 ± 0.000807

Critère : loss de validation, avec préférence pour le direct si son
écart moyen avec les marges est inférieur ou égal à l’écart-type
des différences par seed.

Runs retenus :
- `B_mobilenetv2_frozen_padding_clean_seed42`
- `B_mobilenetv2_frozen_padding_clean_seed43`
- `B_mobilenetv2_frozen_padding_clean_seed44`
