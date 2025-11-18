# Source SDK 2013 — GMOD Tools X64 Mod
## This project is under development, but the GPU/OpenCL optimization is almost working!

So yes, I did it, I created a GPU version of the VVIS tool specifically for GMOD. It uses OpenCL for GPU compatibility, and for a large open map like my Kindercity (it's a big city but with many more things inside), it only took 2 seconds to compile with my GPU (Rtx 4090). So try it out and enjoy this feature. I will compare many maps with the original VVIS and my GPU version to see the difference and what I can optimize. But for now, it works, and I've done the hardest part. Don't be so excited, because like i said, it's ALMOST finished but not actually so you will need to wait quites weeks or months, or just help me :)

### Here all the infos :

Source code for **Source SDK 2013 GMOD (64-bit)** — custom-optimized build tools for **VBSP**, **VVIS**, and **VRAD**, primarily targeting performance improvements, GPU acceleration, and modern compiler compatibility.

This repository is an experimental fork based on [Ficool2’s Source SDK 2013 fork](https://github.com/ficool2/source-sdk-2013), adapted specifically for **Garry’s Mod (64-bit)**.  
It serves as a **research and testing environment**, not a finalized SDK release.

---

## Project Overview

The purpose of this project is to explore how the classic Source Engine build tools can be modernized to take advantage of modern compilers, and even GPU acceleration through **OpenCL**.  

While this repository contains fully compilable tools, it is **not intended for end users** who simply want improved compilers.  
Instead, it aims to document experiments, test optimizations, and demonstrate potential evolutions of the original Source SDK pipeline.

---

## Development Activity Notice

This repository (and others under my account) is currently maintained at a **very slow and irregular pace**.  
Development progress may be **paused for long periods** due to lack of priority and available time.  
Updates, commits, or experimental changes may therefore take **months or even years** before being published.  
The goal remains to preserve and document my work over time, even if active development is infrequent.


## Current Status

### VVIS_GPU (OpenCL)
The GPU-accelerated version of **VVIS** is **currently on hold**.  
With the release of **[VVIS++ and other tools by Ficool2](https://ficool2.github.io/HammerPlusPlus-Website/tools.html)**, CPU-based visibility calculations are already significantly optimized.  
As a result, the OpenCL branch remains an **experimental concept**, to be revisited later for research or hybrid CPU/GPU tests.

### VRAD
Work is in progress to refactor and improve **VRAD**:
- A new **verbose output system** will log details into a separate file.
- Additional **command-line options** are planned for lighting analysis and debugging.
- Potential exploration of **GPU-based bounce lighting**.

### VBSP
Minor profiling and structural tests are being made to analyze compile-time behavior and potential I/O bottlenecks.

---

## Goals

This repository is **not a downloadable SDK** or a ready-to-use replacement for official tools.  
It is meant for developers and enthusiasts who want to:
- Understand how Source tools can be **modernized and optimized**.
- Experiment with **OpenCL integration** in legacy C++ codebases.
- Study compile-time and lighting performance behaviors.
- Follow the evolution of tests, experiments, and refactors over time.

Future updates will also aim to **consolidate all dependencies** into a **single executable** (e.g., `vvis.exe`, `vrad.exe` without external DLLs), ensuring easier compatibility with **Garry’s Mod**.

⚠️ **Note:**  
This project is **Windows-only** for now. Linux support is not native and will be considered later if requested, along with dedicated compatibility explanations or adaptations.

---

## Build Instructions (Windows)

### Requirements
- **Visual Studio 2022** with:
  - Desktop development with C++
  - MSVC v143 build tools
  - Windows 10 or 11 SDK  
- **CMake** (recommended)
- **OpenCL SDK** (Intel, AMD, or NVIDIA supported)

### Building
```bash
git clone https://github.com/<your-username>/SourceSDK2013-GMOD-64Bit.git
cd SourceSDK2013-GMOD-64Bit
cmake -B build -S . -A x64
cmake --build build --config Release
```
Compiled binaries will appear in:

```
/bin/vvis_GPU.exe (Example)
```
### ! Warning !
You will need the subfolder and the dll inside to make tool work : ```/bin/bin/x64/filesystem_stdio.dll```
As for the ```filesystem_stdio.dll```, you will need to keep ```tier0.dll``` and ```vstdlib.dll``` next to the EXEcutable/Release.

## Example of usage :
## Usage of the tool is the same way as the originals ones.

### (VVIS GPU) Global :
```
PathToYourClone\bin\vvis_GPU.exe [SpecificOptions] -game "[PathToGameFolder]" "[PathToBspFile]"
```
### (VVIS GPU) Specific example :
```
.\vvis_GPU.exe -threads 24 -game "C:\Program Files (x86)\Steam\steamapps\common\GarrysMod\garrysmod" "C:\Program Files (x86)\Steam\steamapps\common\GarrysMod\garrysmod\map\gm_kindercity"
```


## References and Resources

- [Valve Developer Wiki — Source SDK 2013](https://developer.valvesoftware.com/wiki/Source_SDK_2013)  
- [Ficool2’s Source SDK 2013 (Base Project)](https://github.com/ficool2/source-sdk-2013)  
- [Ficool2’s Tools (VVIS++, VRAD++, etc.)](https://ficool2.github.io/HammerPlusPlus-Website/tools.html)  
- [OpenCL Specification](https://www.khronos.org/opencl/)  
- [Garry’s Mod Developer Wiki](https://wiki.facepunch.com/gmod/)

---

## License

The SDK and its derivatives are licensed under the **SOURCE 1 SDK LICENSE**, included in the [LICENSE](LICENSE) file.  
All modifications, experiments, and derived GPU or optimization code fall under the same non-commercial license terms.

---

## Author Notes

This repository represents the **beginning of a long-term experimental project** to modernize, analyze, and optimize the Source build tools for Garry’s Mod and beyond.  
The focus is on **research**, **learning**, and **technical curiosity** rather than providing ready-to-use binaries.

> “This is just the beginning of a long journey of experiments and ideas.”
