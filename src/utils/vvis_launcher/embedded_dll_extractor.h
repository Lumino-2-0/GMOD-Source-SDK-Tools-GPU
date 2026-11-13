//========= Copyright Lumastor Corporation, All rights reserved. ============//
//
// Purpose: Extracts a runtime DLL embedded as an RCDATA resource (see
// embedded_dlls.rc / embedded_dlls.h) to a file on disk if it is not
// already present there with the correct size.
//
//=============================================================================//
#ifndef EMBEDDED_DLL_EXTRACTOR_H
#define EMBEDDED_DLL_EXTRACTOR_H
#pragma once

// Extracts the resource identified by resourceId (an RCDATA resource in the
// current executable's own module) to pDestPath. If a file already exists at
// pDestPath with the exact same size as the embedded resource, extraction is
// skipped (idempotent, cheap to call on every run). Returns true on success
// (including the "already up to date" case), false on I/O failure.
bool ExtractEmbeddedDll( int resourceId, const char *pDestPath );

// Returns the directory containing the currently running executable,
// without a trailing slash. Buffer must be at least MAX_PATH.
void GetExecutableDirectory( char *pOutDir, int outDirSize );

#endif // EMBEDDED_DLL_EXTRACTOR_H
