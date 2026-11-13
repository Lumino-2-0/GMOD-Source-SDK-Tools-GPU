# Source SDK 2013 - GMOD Tools GPU (CUDA)

EN [English version](README.md)

VVIS et VRAD pour Garry's Mod (64 bits), avec une partie du calcul envoyée sur le GPU.

## Où en est le projet

Je l'avais dit dans l'ancienne version : je repartais de zéro sur une base **CUDA**, avec un plan
stable. C'est fait. L'ancienne piste OpenCL est abandonnée côté architecture (OpenCL reste là
uniquement comme repli pour VVIS sur AMD/Intel).

Ça reste **expérimental**. Je m'en sers sur mes propres cartes, mais garde une sauvegarde de
tes `.bsp` et vérifie en jeu avant de publier quoi que ce soit.

| Outil | État |
|---|---|
| **VRAD** | Utilisable. Transferts et éclairage des static props sur GPU. |
| **VVIS** | Utilisable mais **pas identique** au VVIS de Valve (voir plus bas). |
| VBSP | Pas touché. |

## Téléchargement

Dans les [Releases](https://github.com/Lumino-2-0/GMOD-Source-SDK-Tools-GPU/releases), ou
directement dans [`bin/`](bin).

Un seul `.exe` par outil : les DLL nécessaires sont embarquées dedans et extraites au premier
lancement dans `%LOCALAPPDATA%`. Garry's Mod est trouvé tout seul via Steam, donc `-game` est
optionnel.

**Il faut :** Windows 64 bits, et pour le GPU une carte NVIDIA RTX 20xx ou plus récente
(compute 7.5). Pas besoin d'installer CUDA, le pilote suffit. Sans GPU compatible, tout repasse
sur le CPU.

## VRAD

```
vrad_gpu.exe -cuda -StaticPropLighting [options VRAD habituelles] ma_carte.bsp
```

| Option | Effet |
|---|---|
| `-cuda` | Active le GPU |
| `-cudaverify` | Calcule CPU **et** GPU et compte les écarts (lent, pour tester) |
| `-NoIndirectGPU` | Laisse l'indirect des static props sur le CPU |

Ce qui part sur le GPU :
- la visibilité des **transferts** entre patches (`MakeAllScales`) ;
- les rayons d'ombre **directs** des static props ;
- l'éclairage **indirect** des static props (un lot par prop).

Sur ma petite carte de test, l'éclairage des props (`.vhv`) sort identique octet par octet
entre CPU et GPU. Pour les transferts, il reste quelques écarts minuscules (0,0024 % des
rayons) : le CPU calcule les inverses avec une approximation SSE, le GPU avec une vraie
division.

Sur `gm_kindercity_night` (ma grosse carte, ~837 000 patches, ~4 700 static props),
RTX 4090 + Ryzen 9 7900X : **10 min 28 s** au total, dont 7 min sur les static props.
Il y a encore de la marge : environ 40 % des sommets de props retombent sur le CPU (surtout à
cause des displacements que le GPU ne sait pas encore traverser). C'est le prochain chantier.

Au passage, j'ai corrigé un crash de VRAD sur les faces avec de très grandes lightmaps
(plus de 32×32 luxels) : le buffer d'échantillons avait une taille fixe trop petite.

## VVIS

```
vvis_gpu.exe [options] ma_carte.bsp
```

| Option | Effet |
|---|---|
| `-cuda` / `-opencl` / `-cpu` | Force le backend (par défaut : CUDA, puis OpenCL, puis CPU) |
| `-final` | Deux passes (4 puis 8 points par anneau), plus complet mais plus lent |
| `-fast` | Pas de raycast, juste le `portalflood` (énorme PVS, très rapide) |
| `-Ray <n>` | Points par anneau de portail (défaut 4, rien ne change au-delà de 8) |
| `-RaycastType 1` | Échantillonnage plus dense (~+25 %) |
| `-Pass 0:Ray=4,1:Ray=8` | Passes à la carte |

Le principe : au lieu du `RecursiveLeafFlow` de Valve, on tire des rayons d'un portail à
l'autre à travers les leafs. Si un rayon passe, la paire est visible. Ça **ajoute** de la
visibilité, ça n'en retire jamais : au pire ça rend un peu trop de choses, jamais un trou
dans la carte.

⚠️ Ce n'est **pas** le même résultat que le VVIS de Valve. Ça marche bien sur des cartes
normales, mais sur des zones pleines de trous (genre une maison en ruine), le PVS peut être
nettement plus large qu'avec le VVIS d'origine.

## Compiler

Visual Studio 2022 (ou plus récent : 2026) (C++ desktop, MSVC, SDK Windows), configuration
**Release x64**. Le toolkit CUDA (`nvcc`) n'est utile que si tu modifies un noyau `.cu`.

**L'ordre compte** : le launcher embarque la DLL au moment où il est compilé. Si tu oublies de
le recompiler, il continue de lancer l'ancienne DLL sans rien dire.

VRAD :
1. Si `vrad_gpu_kernel.cu` a changé : `src/utils/vrad/vrad_build_ptx.ps1`
2. `src/utils/vrad/vrad_dll_win64.vcxproj`
3. `src/utils/vrad_launcher/vrad_launcher_win64.vcxproj`

VVIS :
1. Si `flow_gpu_kernel.cu` a changé : `src/utils/vvis/build_ptx.ps1` (et reporter la modif
   dans `flow_gpu_kernel.h`, la version OpenCL)
2. `src/utils/vvis/vvis_dll_win64.vcxproj`
3. `src/utils/vvis_launcher/vvis_launcher_win64.vcxproj`

## Fichiers intéressants

| Fichier | Rôle |
|---|---|
| `src/utils/vrad/vrad_gpu.cpp` | Côté CPU du GPU pour VRAD (chargement CUDA, lots, files) |
| `src/utils/vrad/vrad_gpu_kernel.cu` | Noyaux VRAD : k-d tree, ombres, indirect |
| `src/utils/vrad/vradstaticprops.cpp` | Éclairage des static props branché sur le GPU |
| `src/utils/vvis/flow_gpu_host.cpp` | Côté CPU du GPU pour VVIS (CUDA + OpenCL, passes) |
| `src/utils/vvis/flow_gpu_kernel.cu` | Noyau VVIS (le principe est expliqué en tête) |
| `src/utils/*_launcher/` | Launchers qui embarquent les DLL |
| `tools/` | Petits scripts Python : PVS d'un point, tailles de lightmap, symboles d'un crash |

## Crédits

- Basé sur le [Source SDK 2013 de ficool2](https://github.com/ficool2/source-sdk-2013).
- [VVIS++ / VRAD++ de ficool2](https://ficool2.github.io/HammerPlusPlus-Website/tools.html),
  qui restent la référence côté CPU.
- [Valve Developer Wiki - Source SDK 2013](https://developer.valvesoftware.com/wiki/Source_SDK_2013)

## Licence

Licence **Source 1 SDK**, voir [LICENSE](LICENSE). Usage non commercial.

---

Le développement avance quand j'ai le temps, entre les études et ma map GMod. Les issues sont
ouvertes, merci de votre patience :)
