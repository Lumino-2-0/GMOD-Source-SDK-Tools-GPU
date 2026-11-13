//========= Copyright Lumastor Corporation, All rights reserved. ============//
//
// Purpose: Resource IDs for the runtime DLLs embedded into vvis_gpu.exe so
// that tier0.dll, vstdlib.dll and filesystem_stdio.dll do not need to be
// shipped as separate files next to the executable.
//
//=============================================================================//
#ifndef EMBEDDED_DLLS_H
#define EMBEDDED_DLLS_H
#pragma once

#define IDR_TIER0_DLL				101
#define IDR_VSTDLIB_DLL				102
#define IDR_FILESYSTEM_STDIO_DLL	103
#define IDR_VVIS_DLL				104

#endif // EMBEDDED_DLLS_H
