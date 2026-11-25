# Tests de non-régression de la qualité d'image

Outil : `veyra_quality.py`. Il répond à une seule question, avec des chiffres : **cette version de Veyra génère-t-elle des images aussi bonnes que la précédente, sur les mêmes scènes ?**

## Principe : retirer une vraie image sur deux

1. On part d'une vraie capture Elgato de GTA 6 (30 images de jeu par seconde).
2. `prepare` enlève les doublons de l'Elgato, garde toutes les vraies images comme **référence**, et fabrique `input.mp4` avec seulement **une vraie image sur deux** (15 i/s).
3. Veyra exporte `input.mp4` en génération d'images 2X. Chaque image générée tombe exactement à la place d'une vraie image retirée.
4. `score` compare chaque image générée à la vraie image retirée.

C'est la méthode standard d'évaluation de l'interpolation d'images (images intermédiaires retenues de Middlebury, Baker et al., IJCV 2011, et triplets Vimeo-90K). Le test est **plus dur** que le direct : le mouvement entre deux images d'entrée est doublé (66 ms au lieu de 33 ms). Il sert à comparer des versions entre elles, pas à prédire la qualité absolue en jeu.

Les images où la PS5 a sauté une image (cadence irrégulière, la vraie image n'est alors pas au milieu) et les changements de plan sont écartés automatiquement.

## Ce qui est mesuré

| Mesure | Ce qu'elle voit | Source |
|---|---|---|
| PSNR, SSIM (luminance BT.709) | Écart global, structure | Wang et al., IEEE TIP 2004 |
| FLIP moyen et p99 | Différences **visibles** quand on alterne les deux images, pensé pour le rendu temps réel | Andersson et al., NVIDIA, HPG 2020 ([code](https://github.com/NVlabs/flip), paquet `flip-evaluator`) |
| FLIP zone personnage | Le perso au centre (zone réglable dans `manifest.json`) | |
| FLIP **personnage immobile** | Les pixels du perso qui ne bougent pas du tout entre les deux vraies images voisines. Un simple mélange 50/50 y est parfait : toute erreur ici, c'est la **tête qui disparaît** ou qui scintille | déduction, voir plus bas |
| FLIP zones immobiles, HUD | HUD traîné par le décor, texte qui bave | masque HUD détecté seul (pixels fixes à bords nets) |
| FLIP bords en mouvement | Les contours qui bougent : halos, doubles contours, trous de désocclusion | |
| Mélange 50/50 | Les mêmes chiffres pour un simple fondu des deux voisines : le plancher que toute génération d'images doit battre | |
| Images répétées | Veyra a recopié une image au lieu d'en générer une | |
| Perte codec | Écart entre les vraies images de la sortie et l'entrée : si c'est bas, un réglage modifie l'image (couleur, SR, débruitage, netteté) | |
| LPIPS (option `--lpips`) | Distance perceptuelle apprise | Zhang et al., CVPR 2018 |

FloLPIPS (Danier et al., [arXiv 2207.08119](https://arxiv.org/abs/2207.08119)) montre que PSNR/SSIM suivent mal ce que voient les gens sur l'interpolation : c'est pourquoi le verdict repose surtout sur FLIP et les zones.

Pour chaque version, les 3 pires images sont enregistrées en PNG (vraie image, image générée, carte d'erreur FLIP, zone perso encadrée) dans `clips/<clip>/results/<version>/`.

### Verdict automatique

La première note d'un clip devient sa référence (`baseline.json`). Ensuite chaque version est comparée à elle. **RÉGRESSION** si une mesure se dégrade au-delà de la marge : FLIP +3 % à +8 % selon la zone, PSNR -0,25 dB, SSIM -0,004. Le code de sortie vaut 1 en cas de régression (utilisable en script). Pour adopter une nouvelle référence : `suite clips --label X --update-baseline`.

## Installation (une fois, Windows)

1. Python 3.10 ou plus récent : `winget install Python.Python.3.12`
2. ffmpeg : `winget install Gyan.FFmpeg`, puis fermer et rouvrir la fenêtre.
3. Dans ce dossier : `py -3 -m pip install -r requirements.txt` (ou lancer `0 - installer.bat`).

## 1. Enregistrer les clips (une fois)

À faire **une seule fois** : ces clips deviennent la banque de test et ne doivent plus changer (sinon les comparaisons entre versions ne veulent plus rien dire).

Réglages d'enregistrement, avec OBS :
- **Fermer Veyra** (l'Elgato ne s'ouvre que dans un logiciel à la fois). On enregistre l'entrée Elgato brute, pas la sortie de Veyra.
- Source : périphérique de capture Elgato, 2560×1440, 60 i/s, aucun filtre ni redimensionnement.
- Sortie > Enregistrement : encodeur **AMD HEVC (AMF)**, contrôle du débit **CQP**, **QP 16**, intervalle d'images clés 1 s, format MKV. (Ou qualité « Sans perte » si l'espace disque le permet.)
- Avancé > Vidéo : espace colorimétrique Rec. 709, plage **Limitée**, comme la chaîne PS5 → Veyra.
- PS5 : mêmes réglages qu'en jeu (Plage RGB Limitée, pas de VRR sur l'entrée Elgato).

Clips à faire, **12 à 15 secondes chacun**, mouvement régulier sans pause :

| Nom du clip | Quoi faire en jeu |
|---|---|
| `tete-rotation` | Perso immobile, tourner la caméra autour de lui au stick droit à vitesse constante. C'est le cas de la tête qui disparaît. |
| `marche-camera` | Marcher en tournant la caméra. |
| `hud` | Caméra qui bouge avec la mini-carte et des indicateurs bien visibles (idéalement sous-titres ou notification). |
| `feuillage` | Herbes, arbres, végétation avec du vent, caméra qui bouge. |
| `nuit` | Scène sombre avec néons et phares. |
| `vehicule-rapide` | Conduite rapide, mouvement fort. |

Ensuite, glisser chaque enregistrement sur `1 - preparer un clip.bat` et taper son nom. Ou en ligne de commande :

```
py -3 veyra_quality.py prepare capture.mkv --name tete-rotation --start 2 --duration 12 --tag tete
```

`--start` saute le début (le temps de lancer le mouvement). Si le personnage n'est pas dans la zone par défaut (centre, de 36 % à 64 % en largeur et 28 % à 92 % en hauteur), corriger `head_rect` dans `clips/<clip>/manifest.json`. Des rectangles HUD peuvent aussi y être ajoutés (`hud_rects`).

## 2. À chaque version de Veyra à tester

1. Lancer Veyra avec le .bat de test voulu (les interrupteurs `VEYRA_TEST_*` s'appliquent aussi à l'export).
2. Réglages : génération d'images **FSR 2X** (XeSS ne fournit pas ses images générées à l'encodeur, l'export le refuse), flux optique AMD FidelityFX, **Edge Aegis activé**. **Désactiver** super-résolution, débruitage, look couleur et la netteté RCAS (`VEYRA_TEST_RCAS`) : ils modifient aussi les vraies images et faussent la comparaison. Débit d'export élevé.
3. Page Export : ajouter `clips\<clip>\input.mp4`, exporter en **`clips\<clip>\veyra_out.mp4`**. Recommencer pour chaque clip.
4. Lancer `2 - noter tous les clips.bat` et donner un nom de version (ex. `2.2.1-flow-v3`). Ou :

```
py -3 veyra_quality.py suite clips --label 2.2.1-flow-v3
```

Le résumé arrive dans `clips/summary_<version>.md`, le détail par image dans chaque `result.json`.

**Ce que le test couvre** : tout ce que Veyra calcule avant la synthèse, c'est-à-dire les vecteurs de mouvement envoyés à la génération (flux optique AMD, Edge Aegis, FlowRefine, médiane, couches, mouvement caméra, protection du HUD). C'est là que naissait la tête qui disparaît (elle ne disparaît pas avec le flux coupé). **Ce qu'il ne couvre pas** : la synthèse propre à XeSS-FG, qui n'est pas exportable. Une version qui gagne ici avec FSR doit encore être confirmée en jeu avec XeSS.

Durée : environ 1 à 3 s par image générée et par processeur à 1440p (FLIP tourne sur le CPU, il ne prend pas de temps au GPU). Le mélange 50/50 est calculé une seule fois par clip puis mis en cache. `--scale 0.5` divise le temps par 4 environ, `--every 2` ne teste qu'une image sur deux.

## Autres commandes

- `score <clip> <export.mp4> --label X` : noter un seul export.
- `gate <result.json> <baseline.json>` : comparer deux notes du même clip.
- `selftest` : vérifie l'outil sur une vidéo synthétique (perso immobile, décor qui défile, HUD). Lancé en CI à chaque modification de ce dossier (`.github/workflows/quality-tool-checks.yml`).

## Limites connues

- Le mouvement est doublé par rapport au direct : les scores absolus sont plus bas qu'en jeu.
- Le personnage n'est repéré que par une zone fixe à l'écran (bonne approximation en vue à la 3e personne, fausse en cinématique).
- FLIP suppose par défaut 67 pixels par degré (`--ppd`). Pour un écran 27" 1440p vu à 60 cm, environ 45 est plus proche. Garder la même valeur pour toutes les comparaisons.
- Le masque HUD automatique est coupé si la caméra bouge trop peu dans le clip.
