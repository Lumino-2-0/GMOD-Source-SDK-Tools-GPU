# Source SDK 2013 — Outils GMOD (Optimisés 64-bit)

Code source du **Source SDK 2013 GMOD (64 bits)** — outils VBSP, VVIS et VRAD personnalisés et optimisés, visant principalement à l'expérimentation, aux tests de performance et à l’intégration GPU via **OpenCL**.

Ce dépôt est basé sur [le fork Source SDK 2013 de Ficool2](https://github.com/ficool2/source-sdk-2013) et adapté spécifiquement pour **Garry’s Mod (64-bit)**.  
Il sert de **tests et d’expérimentations**, et n’est pas une version finale ou prête à l’usage.

---

## Présentation du projet

L’objectif est d’explorer comment les outils de compilation classiques du moteur Source peuvent être modernisés pour tirer parti des processeurs multicœurs, des compilateurs récents et même du GPU avec **OpenCL**.  

Ce dépôt contient des outils compilables, mais il n’est **pas destiné aux utilisateurs cherchant des compilateurs prêts à l’emploi**.  
Il documente les expérimentations, les optimisations et démontre ce qui est possible avec la chaîne de compilation Source SDK.

---

## État actuel

### VVIS_GPU (OpenCL)
La version GPU de **VVIS** est **actuellement en pause**.  
Avec la sortie de **[VVIS++ et autres outils de Ficool2](https://ficool2.github.io/HammerPlusPlus-Website/tools.html)**, les calculs CPU sont déjà très optimisés.  
Le développement OpenCL reste une **expérimentation** et pourra être repris plus tard pour des tests hybrides CPU/GPU.

### VRAD
Travail en cours sur **VRAD** :
- Nouveau système de **verbose** avec logs dans un fichier séparé.
- Options supplémentaires en ligne de commande pour le débogage et l’analyse d’éclairage.
- Expérimentations potentielles sur le calcul d’éclairage indirect GPU.

### VBSP
Profiling et tests internes pour analyser les performances et les goulots d’étranglement possibles dans le traitement des brushes et entités.

---

## Objectifs

Ce dépôt **n’est pas un SDK téléchargeable** ou un remplacement des outils officiels.  
Il sert à :
- Comprendre comment les outils Source peuvent être **modernisés et optimisés**.
- Expérimenter l’**intégration GPU** dans un code C++ ancien.
- Étudier les performances de compilation et d’éclairage.
- Suivre l’évolution des tests et des expérimentations dans le temps.

À terme, le projet vise à **regrouper toutes les dépendances dans un seul exécutable** (`vvis.exe`, `vrad.exe`) sans DLL externes pour assurer une compatibilité simple avec **Garry’s Mod**.

⚠️ **Note :**  
Le projet est pour l’instant **Windows uniquement**. Le support Linux n’est pas natif et pourra être étudié ultérieurement si besoin.

---

## Compilation sous Windows

### Prérequis
- **Visual Studio 2022** avec :
  - Développement C++ Desktop
  - Outils MSVC v143
  - SDK Windows 10 ou 11  
- **CMake** (recommandé)
- **OpenCL SDK** (Intel / AMD / NVIDIA)

### Commandes
```bash
git clone https://github.com/<votre-utilisateur>/SourceSDK2013-GMOD-64Bit.git
cd SourceSDK2013-GMOD-64Bit
cmake -B build -S . -A x64
cmake --build build --config Release
```

Les exécutables compilés se trouveront dans :

```
build/game/bin/
```
## Ressources utiles

- [Valve Developer Wiki — Source SDK 2013](https://developer.valvesoftware.com/wiki/Source_SDK_2013)  
- [Source SDK 2013 de Ficool2](https://github.com/ficool2/source-sdk-2013)  
- [Outils Ficool2 : VVIS++, VRAD++, etc.](https://ficool2.github.io/HammerPlusPlus-Website/tools.html)  
- [Spécification OpenCL](https://www.khronos.org/opencl/)  
- [Wiki Développeur Garry’s Mod](https://wiki.facepunch.com/gmod/)

---

## Licence

Le SDK et ses dérivés sont sous **SOURCE 1 SDK LICENSE**, inclus dans le fichier [LICENSE](LICENSE).  
Toutes les modifications expérimentales, optimisations ou intégrations GPU sont soumises aux mêmes conditions non commerciales.

---

## Avis sur l’activité du projet

Ce dépôt est **actuellement maintenu très lentement et de manière irrégulière**.  
Les développements, commits ou expérimentations peuvent prendre **des mois, voire des années** avant d’être publiés, faute de priorité.  
L’objectif est de **préserver et documenter le travail**, même si le développement actif est peu fréquent.

---

## Notes de l’auteur

Ce dépôt représente le **début d’un long projet d’expérimentations** pour moderniser, analyser et optimiser les outils de compilation Source pour Garry’s Mod.  
L’accent est mis sur la **recherche, l’apprentissage et la curiosité technique**, plutôt que sur la livraison de binaires prêts à l’usage.

> “Ce n’est que le commencement d’une longue série d’expériences et d’idées.”

