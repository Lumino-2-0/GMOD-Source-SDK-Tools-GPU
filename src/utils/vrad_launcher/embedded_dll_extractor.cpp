//========= Copyright Lumastor Corporation, All rights reserved. ============//
//
// Purpose: Implementation of embedded DLL extraction. See
// embedded_dll_extractor.h for the rationale.
//
//=============================================================================//
#include <windows.h>
#include <stdio.h>
#include "embedded_dll_extractor.h"

void GetExecutableDirectory( char *pOutDir, int outDirSize )
{
	pOutDir[0] = 0;

	char szExePath[MAX_PATH];
	DWORD len = GetModuleFileNameA( NULL, szExePath, MAX_PATH );
	if ( len == 0 || len >= MAX_PATH )
		return;

	// Strip the filename, keep the directory.
	char *pLastSlash = strrchr( szExePath, '\\' );
	if ( !pLastSlash )
		pLastSlash = strrchr( szExePath, '/' );

	if ( pLastSlash )
		*pLastSlash = 0;

	strncpy_s( pOutDir, outDirSize, szExePath, _TRUNCATE );
}

static bool CreateDirectoryRecursive( const char *pPath )
{
	char szTemp[MAX_PATH];
	strncpy_s( szTemp, pPath, _TRUNCATE );

	for ( char *p = szTemp + 1; *p; ++p )
	{
		if ( *p == '\\' || *p == '/' )
		{
			char c = *p;
			*p = 0;
			CreateDirectoryA( szTemp, NULL ); // ignore failure (may already exist)
			*p = c;
		}
	}

	return CreateDirectoryA( szTemp, NULL ) || GetLastError() == ERROR_ALREADY_EXISTS;
}

bool ExtractEmbeddedDll( int resourceId, const char *pDestPath )
{
	HMODULE hSelf = GetModuleHandleA( NULL );

	HRSRC hRes = FindResourceA( hSelf, MAKEINTRESOURCEA( resourceId ), RT_RCDATA );
	if ( !hRes )
		return false;

	DWORD resSize = SizeofResource( hSelf, hRes );
	if ( resSize == 0 )
		return false;

	HGLOBAL hResData = LoadResource( hSelf, hRes );
	if ( !hResData )
		return false;

	const void *pResBytes = LockResource( hResData );
	if ( !pResBytes )
		return false;

	// If the destination file already exists with the exact expected size,
	// assume it is up to date and skip the (redundant) write. This keeps
	// repeated launches cheap and avoids "file in use" issues once the DLL
	// has already been loaded by a previous, still-running instance.
	WIN32_FILE_ATTRIBUTE_DATA existingAttrs;
	if ( GetFileAttributesExA( pDestPath, GetFileExInfoStandard, &existingAttrs ) )
	{
		LARGE_INTEGER existingSize;
		existingSize.HighPart = existingAttrs.nFileSizeHigh;
		existingSize.LowPart = existingAttrs.nFileSizeLow;
		if ( (DWORD)existingSize.QuadPart == resSize )
			return true;
	}

	// Make sure the destination directory exists (e.g. "bin\x64").
	char szDir[MAX_PATH];
	strncpy_s( szDir, pDestPath, _TRUNCATE );
	char *pLastSlash = strrchr( szDir, '\\' );
	if ( !pLastSlash )
		pLastSlash = strrchr( szDir, '/' );
	if ( pLastSlash )
	{
		*pLastSlash = 0;
		CreateDirectoryRecursive( szDir );
	}

	// Write to a temp file first, then atomically rename into place. This
	// avoids leaving a truncated DLL behind if two instances race, and
	// avoids failing outright if the target is currently mapped/locked by
	// another already-running instance (MoveFileEx below will just no-op
	// in that case since the existing file matches by size on next launch).
	char szTempPath[MAX_PATH];
	_snprintf_s( szTempPath, _TRUNCATE, "%s.tmp%lu", pDestPath, GetCurrentProcessId() );

	HANDLE hFile = CreateFileA( szTempPath, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL );
	if ( hFile == INVALID_HANDLE_VALUE )
		return false;

	DWORD bytesWritten = 0;
	BOOL bOk = WriteFile( hFile, pResBytes, resSize, &bytesWritten, NULL );
	CloseHandle( hFile );

	if ( !bOk || bytesWritten != resSize )
	{
		DeleteFileA( szTempPath );
		return false;
	}

	if ( !MoveFileExA( szTempPath, pDestPath, MOVEFILE_REPLACE_EXISTING ) )
	{
		// Destination might be locked by another running instance that
		// already loaded it; that's fine, it's already usable as-is.
		DeleteFileA( szTempPath );
		WIN32_FILE_ATTRIBUTE_DATA attrs;
		if ( GetFileAttributesExA( pDestPath, GetFileExInfoStandard, &attrs ) )
		{
			LARGE_INTEGER sz;
			sz.HighPart = attrs.nFileSizeHigh;
			sz.LowPart = attrs.nFileSizeLow;
			return (DWORD)sz.QuadPart == resSize;
		}
		return false;
	}

	return true;
}
