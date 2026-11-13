# Source SDK 2013 - GMOD Tools GPU (CUDA+OpenCL)

FR [Version française](README_FR.md)

VVIS and VRAD for Garry's Mod (64-bit), with part of the work moved to the GPU.

## Project status

I said it in the old version: I was starting over on a **CUDA** base with a stable plan. Done.
The old OpenCL direction is dropped as an architecture (OpenCL is only kept as a fallback for
VVIS on AMD/Intel GPU).

It's still **experimental**. I use it on my own maps, but keep a backup of your `.bsp` files
and check in game before publishing anything.

| Tool | Status |
|---|---|
| **VRAD** | Usable. Transfers and static prop lighting on GPU. |
| **VVIS** | Usable but **not identical** to Valve's VVIS (see below). |
| VBSP | Untouched. |

## Download

From the [Releases](https://github.com/Lumino-2-0/GMOD-Source-SDK-Tools-GPU/releases), or
straight from [`bin/`](bin).

One `.exe` per tool: the required DLLs are embedded and extracted on first launch into
`%LOCALAPPDATA%`. Garry's Mod is found automatically through Steam, so `-game` is optional.

**You need:** 64-bit Windows, and for the GPU path an NVIDIA RTX 20xx or newer (compute 7.5).
No CUDA install needed, the driver is enough. Without a compatible GPU, everything falls back
to the CPU.

## VRAD

```
vrad_gpu.exe -cuda -StaticPropLighting [usual VRAD options] my_map.bsp
```

| Option | Effect |
|---|---|
| `-cuda` | Enable the GPU |
| `-cudaverify` | Run CPU **and** GPU and count mismatches (slow, for testing) |
| `-NoIndirectGPU` | Keep static prop indirect lighting on the CPU |

What runs on the GPU:
- patch-to-patch **transfer** visibility (`MakeAllScales`);
- **direct** shadow rays for static props;
- **indirect** lighting for static props (one batch per prop).

On my small test map, prop lighting (`.vhv`) comes out byte-for-byte identical between CPU and
GPU. Transfers still have a few tiny differences (0.0024% of rays): the CPU computes
reciprocals with an SSE approximation, the GPU with a real division.

On `gm_kindercity_night` (my big map, ~837,000 patches, ~4,700 static props), RTX 4090 +
Ryzen 9 7900X: **10 min 28 s** total, 7 min of that on static props. There's still room:
about 40% of prop vertices fall back to the CPU (mostly because of displacements the GPU can't
traverse yet). That's the next thing on the list.

Along the way I fixed a VRAD crash on faces with very large lightmaps (over 32×32 luxels): the
sample buffer had a fixed size that was too small.

## VVIS

```
vvis_gpu.exe [options] my_map.bsp
```

| Option | Effect |
|---|---|
| `-cuda` / `-opencl` / `-cpu` | Force the backend (default: CUDA, then OpenCL, then CPU) |
| `-final` | Two passes (4 then 8 points per ring), more complete but slower |
| `-fast` | No raycast, just `portalflood` (huge PVS, very fast) |
| `-Ray <n>` | Points per portal ring (default 4, nothing changes past 8) |
| `-RaycastType 1` | Denser sampling (~+25%) |
| `-Pass 0:Ray=4,1:Ray=8` | Custom passes |

The idea: instead of Valve's `RecursiveLeafFlow`, rays are shot from one portal to another
through the leafs. If a ray gets through, the pair is visible. It only **adds** visibility,
never removes it: worst case you render a bit too much, never a hole in the map.

⚠️ It is **not** the same result as Valve's VVIS. It works fine on normal maps, but in areas
full of holes (think a ruined house), the PVS can be noticeably larger than with the original
VVIS.

## Building

Visual Studio 2022 or newer (C++ desktop, MSVC, Windows SDK), **Release x64**. The CUDA toolkit
(`nvcc`) is only needed if you edit a `.cu` kernel.

**Order matters**: the launcher embeds the DLL at the time it's built. If you forget to rebuild
it, it keeps running the old DLL without a word.

VRAD:
1. If `vrad_gpu_kernel.cu` changed: `src/utils/vrad/vrad_build_ptx.ps1`
2. `src/utils/vrad/vrad_dll_win64.vcxproj`
3. `src/utils/vrad_launcher/vrad_launcher_win64.vcxproj`

VVIS:
1. If `flow_gpu_kernel.cu` changed: `src/utils/vvis/build_ptx.ps1` (and mirror the change in
   `flow_gpu_kernel.h`, the OpenCL version)
2. `src/utils/vvis/vvis_dll_win64.vcxproj`
3. `src/utils/vvis_launcher/vvis_launcher_win64.vcxproj`

## Interesting files

| File | Role |
|---|---|
| `src/utils/vrad/vrad_gpu.cpp` | Host side of VRAD's GPU code (CUDA loading, batches, queues) |
| `src/utils/vrad/vrad_gpu_kernel.cu` | VRAD kernels: k-d tree, shadows, indirect |
| `src/utils/vrad/vradstaticprops.cpp` | Static prop lighting wired to the GPU |
| `src/utils/vvis/flow_gpu_host.cpp` | Host side of VVIS's GPU code (CUDA + OpenCL, passes) |
| `src/utils/vvis/flow_gpu_kernel.cu` | VVIS kernel (the idea is explained at the top) |
| `src/utils/*_launcher/` | Launchers embedding the DLLs |
| `tools/` | Small Python scripts: PVS from a point, lightmap sizes, crash symbols |

## Credits

- Based on [ficool2's Source SDK 2013](https://github.com/ficool2/source-sdk-2013).
- [ficool2's VVIS++ / VRAD++](https://ficool2.github.io/HammerPlusPlus-Website/tools.html),
  still the reference on the CPU side.
- [Valve Developer Wiki - Source SDK 2013](https://developer.valvesoftware.com/wiki/Source_SDK_2013)

## License

**Source 1 SDK** license, see [LICENSE](LICENSE). Non-commercial use.

---

Development moves when I have time, between school and my GMod map. Issues are open, thanks
for your patience :)
