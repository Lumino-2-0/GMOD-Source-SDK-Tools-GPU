# Source SDK 2013 - GMOD Tools GPU (CUDA+OpenCL)

EN [English version](README.md)

VVIS et VRAD pour Garry's Mod (64 bits), avec une partie du calcul envoyée sur le GPU.

## En construction, mises à jours fréquentes !

## Où en est le projet

Je l'avais dit dans l'ancienne version : je repartais de zéro sur une base **CUDA**, avec un plan
stable. C'est fait. L'ancienne piste OpenCL est abandonnée côté architecture (OpenCL reste là
uniquement comme repli pour VVIS sur les GPU AMD/Intel).

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
| (défaut) | Raycast sur **tous les moteurs à la fois** : le GPU principal, les autres GPU (GPU intégré, via OpenCL) et les threads CPU (AVX2 si dispo) |
| `-gpuonly` | GPU seulement (pas de threads CPU) |
| `-nogpu` | Threads CPU seulement (même PVS, plus lent) |
| `-cuda` / `-opencl` | API du GPU principal (par défaut : CUDA sur NVIDIA, sinon OpenCL) |
| `-cpu` | Le flow CPU récursif d'origine de Valve (pas de raycast) |
| `-fast` | Pas de raycast, juste le `portalflood` (énorme PVS, très rapide) |
| `-final`, `-Ray`, `-RaycastType`, `-Pass` | Acceptées mais ignorées : la précision est fixe (voir plus bas) |
| `-debug` | Sortie GPU détaillée (moteurs, chaque passe, paires candidates, répartition du travail) |
| `-nocolor` | Sortie en texte simple (aussi `NO_COLOR=1` ; `FORCE_COLOR=1` garde les couleurs si la sortie est redirigée) |
| `-noupdate` | Pas de vérification de mise à jour au démarrage (aussi `VVIS_GPU_NOUPDATE=1`) |

Le principe : au lieu du `RecursiveLeafFlow` de Valve, on tire des rayons d'un portail à
l'autre à travers les leafs. Si un rayon passe, la paire est visible. Par défaut il y a deux
passes :

1. **Anneaux** : le centre du portail plus des anneaux vers ses coins, on s'arrête au premier
   rayon qui passe.
2. **Échantillonnage dense** sur les paires que la première passe n'a pas pu prouver, en
   tirant tous les rayons. Le nombre de points (jusqu'à 128 par portail) vient d'un budget de
   travail qui ne dépend que de la carte (au moins 600 M rayons, ou 1/8 de la première
   passe) : complet sur les petites cartes, léger ou sauté sur les énormes, donc le temps de
   compilation reste proche de celui de la première passe.

Les rayons ne peuvent pas frôler les bords des portails, les points restent légèrement à
l'intérieur des coins, et le résultat est limité au `portalflood` : un rayon ne peut plus
« fuir » par le coin d'un mur.

**Le même PVS partout.** Rien ne dépend de la vitesse de la machine : les rayons sont tirés par
groupes fixes de 32 et tous les moteurs (CUDA, OpenCL, CPU SSE2/AVX2) exécutent le même code,
donc une carte donne un PVS identique à l'octet près sur n'importe quel PC, quel que soit le
mélange de moteurs. C'est pour ça que `-final` a disparu : il n'y a qu'une précision, la
meilleure qu'on peut se permettre.

**Tous les moteurs ensemble.** Le travail est découpé en petits lots que les GPU et les threads
CPU prennent dès qu'ils ont fini, donc un GPU rapide fait l'essentiel et personne n'attend. Le
GPU répartit les rayons de chaque paire sur un warp entier (et sur plusieurs warps pendant la
passe dense), et garde deux lots en vol, donc sa charge reste à 100 %.

Comparé au VVIS de Valve (`-cpu`), sur mes cartes de test le GPU ne rend jamais visible une
paire que le CPU cache, et il n'en rate que quelques-unes derrière des ouvertures très étroites
(paires de clusters manquantes par rapport au CPU) :

| Carte | v0.1 | v0.2 |
|---|---|---|
| `campaign_1` | 92 | 8 |
| `gm_teste_vvis` | 2968 | 806 |

Sur `gm_kindercity_night` (7 019 clusters, RTX 4090 + Ryzen 9 7900X), le raycast prend environ
6 s et tout VVIS environ 10 s (environ 43 s pour la v0.1). Si tu vois un trou dans la carte,
ouvre une issue avec la position (`getpos`).

À la fin, tous les modes (GPU, `-cpu`, `-fast`) affichent la même ligne `PVS final: x%.` : la
part de la carte visible depuis un cluster moyen (plus c'est bas, plus on cache). Sur
`campaign_1` : GPU 41,96 %, CPU 42,11 %, `-fast` 65,78 %. La ligne CPU
`Optimized: N visible clusters (x%)` compte les liens à sens unique retirés par la
vérification finale, pas la taille du PVS : le GPU ne l'affiche qu'avec `-debug`.

La console est en couleurs, avec une bannière en ASCII art et une seule barre de progression
`PortalFlow_RayGPU` (texte simple si la sortie est redirigée, et dans le `.log`).

**Mises à jour :** au démarrage, `vvis_gpu.exe` regarde les
[tags GitHub](https://github.com/Lumino-2-0/GMOD-Source-SDK-Tools-GPU/tags). Si un tag est plus
récent que l'exe et que sa release a un fichier à télécharger, il le télécharge, se remplace
(l'ancien exe est gardé en `vvis_gpu.exe.old` jusqu'au lancement suivant) et se relance avec
les mêmes arguments. Sans réseau, ou si l'exe est dans un dossier où il ne peut pas écrire, il
garde simplement la version actuelle.
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
| `src/utils/vvis/flow_gpu_host.cpp` | Pilotage du raycast VVIS (moteurs, répartition, passes) |
| `src/utils/vvis/flow_gpu_kernel.cu` | Noyau VVIS, commun à CUDA et au CPU (le principe est expliqué en tête) |
| `src/utils/vvis/flow_cpu_ray*.cpp` | Le même noyau compilé pour les threads CPU (SSE2 et AVX2) |
| `src/utils/vvis/vvis_ui.cpp` | Sortie console de VVIS (couleurs, bannière, barre de progression) |
| `src/utils/vvis_launcher/vvis_updater.cpp` | Vérification et installation des mises à jour depuis GitHub |
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
