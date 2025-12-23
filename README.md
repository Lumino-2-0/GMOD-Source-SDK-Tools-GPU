# Source SDK 2013 — GMOD Tools X64 Mod

# DO NOT USE YET — CPU BASELINE REBUILD IN PROGRESS

---

##  Project Status — IMPORTANT

This project is currently undergoing a **full reset of the VVIS tool**.

After extensive experimentation, it became clear that:
- The **CPU VVIS baseline must be 100% identical to Valve’s original behavior**
- Any GPU or OpenCL optimization **must be layered strictly on top of a verified CPU reference**
- Even small deviations in PortalFlow, cluster merge, or compression logic can produce **incorrect PVS results**

**As of now, GPU acceleration is TEMPORARILY DISABLED**  
**VVIS is being rebuilt from a clean, Valve-faithful baseline**

**Do NOT use this repository for production maps at this time.**

---

## Project Summary

Source code for **Source SDK 2013 (Garry’s Mod, 64-bit)** — custom experimental build tools for:

- **VBSP**
- **VVIS**
- **VRAD**

The long-term goal is to explore **modernization, profiling, and acceleration** of Source Engine build tools while preserving:
- Perfect format compatibility
- Deterministic results
- Zero visual regressions

This repository is an **experimental research project**, not a drop-in replacement for official tools.

---

## Background & Motivation

This fork is based on **Source SDK 2013**, originally adapted from  
[Ficool2’s Source SDK 2013 fork](https://github.com/ficool2/source-sdk-2013),  
and modified specifically for **Garry’s Mod (64-bit)**.

The original ambition of this project was to:
- Explore **GPU-assisted visibility computation (VVIS)**
- Use **OpenCL** for broad GPU compatibility
- Reduce compile times on large, complex maps
- Preserve **strict correctness guarantees**

However, practical experimentation revealed that:
> **VVIS is extremely sensitive to implementation details**  
> Any optimization is meaningless unless the CPU version is first proven identical to Valve’s.

---


### GPU / OpenCL Work — ON HOLD

The OpenCL-based **VVIS_GPU** work is currently **paused**, not abandoned.

Reasons:
- CPU correctness has absolute priority
- Ficool2’s **VVIS++** already provides excellent CPU-side performance
- GPU acceleration must never compromise correctness

GPU work will resume only when:
- CPU baseline matches Valve exactly
- All differences are fully understood and documented
- GPU is used only for **safe, conservative pruning**

---

### VRAD — PLANNED IMPROVEMENTS

Future work on **VRAD** may include:
- Improved verbose logging (separate log files)
- Additional debug and analysis flags
- Exploration of **GPU-assisted bounce lighting** (research only)

No guarantees or timelines.

---

###  VBSP

Minor profiling and structural analysis only:
- Compile-time behavior
- I/O bottlenecks
- No functional changes planned for now

- Upgrades limits ?
  
---

## Development Activity Notice

Development is **slow, irregular, and priority-based**.

There may be:
- Long periods without commits
- Experimental branches that are later discarded
- Major refactors with no short-term payoff

The goal is to **preserve knowledge and experiments**, not to rush releases.

---

## Platform Support

- Windows only (for now)
- Linux not supported (may be explored later)

---

## Build Instructions (Windows)

### Requirements

- **Visual Studio 2022**
  - Desktop development with C++
  - MSVC v143
  - Windows 10 / 11 SDK
- **CMake** (recommended)
- **OpenCL SDK** (optional, currently unused)

### Build Notes

- Use **Release** configuration (**NOT Debug**)
- Debugging can still be done via *Local Windows Debugger*
- Tools are built as standalone executables

Compiled binaries will appear in:
```
/bin/
```

Example:
```
/bin/vvis_GPU.exe
```

### Runtime Dependencies

You must keep the following next to the executable:
```
filesystem_stdio.dll (in the subfolders : /bin/bin/x64/)
tier0.dll
vstdlib.dll
```

Directory layout must be preserved.

---

## Tool Usage

Usage syntax remains identical to the original Source tools.

Example:
```
PathToYourClone\bin\vvis_GPU.exe -game "PathToGarrysMod\garrysmod" "PathToMap\map.bsp"
```

---

## References & Resources

- Valve Developer Wiki — Source SDK 2013  
  https://developer.valvesoftware.com/wiki/Source_SDK_2013

- Ficool2 — Source SDK 2013 fork  
  https://github.com/ficool2/source-sdk-2013

- Ficool2 — VVIS++, VRAD++, Tools  
  https://ficool2.github.io/HammerPlusPlus-Website/tools.html

- OpenCL Specification  
  https://www.khronos.org/opencl/

- Garry’s Mod Developer Wiki  
  https://wiki.facepunch.com/gmod/

---

## License

This project is licensed under the **SOURCE 1 SDK LICENSE**.  
See the [LICENSE](LICENSE) file for details.

All derived work, experiments, and modifications remain subject to the same non-commercial license.

---

## Author Notes

This repository reflects an **honest, technical exploration** of Source Engine tooling.

Progress is nonlinear.  
Mistakes are part of the process.  
Correctness always comes before speed.

> “Make it correct.  
> Then make it fast.  
> Never the other way around.”
