//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: Single-executable bootstrap for VVIS GPU.
//
// vvis_gpu.exe embarque tier0, vstdlib, filesystem_stdio et vvis_dll en
// RCDATA (embedded_dlls.rc). Au lancement on les extrait dans
// %LOCALAPPDATA%\vvis_gpu\<hash>, puis on charge vvis_dll.dll comme le
// vvis_launcher d'origine. Un seul exe a distribuer.
//
// GMod est detecte via Steam (registre + libraryfolders.vdf) : -game est
// ajoute tout seul si absent. -gamebin = utiliser les DLL x64 du jeu.
//
// Pas de lien vers tier0/vstdlib ici (tier0 exporte des donnees, donc pas
// de /DELAYLOAD possible) : uniquement du Win32 brut LoadLibrary/GetProcAddress.
//
//=============================================================================//

#include "stdafx.h"
#include <stdio.h>
#include <malloc.h>
#include "ilaunchabledll.h"
#include "embedded_dlls.h"
#include "embedded_dll_extractor.h"

typedef void* (*CreateInterfaceFn)( const char *pName, int *pReturnCode );

static char* GetLastErrorString()
{
	static char err[2048];

	LPVOID lpMsgBuf;
	FormatMessage(
		FORMAT_MESSAGE_ALLOCATE_BUFFER |
		FORMAT_MESSAGE_FROM_SYSTEM |
		FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL,
		GetLastError(),
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
		(LPTSTR)&lpMsgBuf,
		0,
		NULL
	);

	strncpy_s( err, (char*)lpMsgBuf, _TRUNCATE );
	LocalFree( lpMsgBuf );

	return err;
}

static bool FileExists( const char *pPath )
{
	DWORD attrs = GetFileAttributesA( pPath );
	return attrs != INVALID_FILE_ATTRIBUTES && !( attrs & FILE_ATTRIBUTE_DIRECTORY );
}

// FNV-1a hash of an embedded resource. Used to name the private extraction
// directory so that a rebuilt launcher never reuses a stale extracted DLL
// (the extractor only compares file sizes).
static unsigned int HashEmbeddedResource( int resourceId )
{
	HMODULE hSelf = GetModuleHandleA( NULL );
	HRSRC hRes = FindResourceA( hSelf, MAKEINTRESOURCEA( resourceId ), RT_RCDATA );
	if ( !hRes )
		return 0;
	DWORD size = SizeofResource( hSelf, hRes );
	HGLOBAL hData = LoadResource( hSelf, hRes );
	const unsigned char *p = hData ? (const unsigned char *)LockResource( hData ) : NULL;
	if ( !p )
		return 0;
	unsigned int h = 2166136261u;
	for ( DWORD i = 0; i < size; ++i )
	{
		h ^= p[i];
		h *= 16777619u;
	}
	return h;
}

// Private runtime directory: %LOCALAPPDATA%\vvis_gpu\<hash>. Keeps the folder
// containing vvis_gpu.exe clean (and allows read-only locations). Falls back
// to the executable directory if LOCALAPPDATA is unavailable.
static void GetRuntimeDirectory( char *pOutDir, int outDirSize )
{
	char szBase[MAX_PATH];
	DWORD len = GetEnvironmentVariableA( "LOCALAPPDATA", szBase, sizeof( szBase ) );
	if ( len == 0 || len >= sizeof( szBase ) )
	{
		GetExecutableDirectory( pOutDir, outDirSize );
		return;
	}
	unsigned int h = HashEmbeddedResource( IDR_VVIS_DLL ) ^ ( HashEmbeddedResource( IDR_TIER0_DLL ) * 31u ) ^ ( HashEmbeddedResource( IDR_FILESYSTEM_STDIO_DLL ) * 131u );
	_snprintf_s( pOutDir, outDirSize, _TRUNCATE, "%s\\vvis_gpu\\%08x", szBase, h );
}

// Returns true if the installed game's 64-bit runtime DLL is a PE image for
// AMD64 (the launcher and vvis_dll.dll are x64; 32-bit game DLLs can't load).
static bool IsX64Dll( const char *pPath )
{
	HANDLE hFile = CreateFileA( pPath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL );
	if ( hFile == INVALID_HANDLE_VALUE )
		return false;
	IMAGE_DOS_HEADER dos;
	IMAGE_NT_HEADERS64 nt;
	DWORD rd = 0;
	bool ok = ReadFile( hFile, &dos, sizeof( dos ), &rd, NULL ) && rd == sizeof( dos ) && dos.e_magic == IMAGE_DOS_SIGNATURE
		&& SetFilePointer( hFile, dos.e_lfanew, NULL, FILE_BEGIN ) != INVALID_SET_FILE_POINTER
		&& ReadFile( hFile, &nt, sizeof( nt ), &rd, NULL ) && rd == sizeof( nt )
		&& nt.Signature == IMAGE_NT_SIGNATURE && nt.FileHeader.Machine == IMAGE_FILE_MACHINE_AMD64;
	CloseHandle( hFile );
	return ok;
}

static bool ReadRegString( HKEY root, const char *pKey, const char *pValue, char *pOut, DWORD outSize )
{
	DWORD type = 0;
	DWORD size = outSize;
	if ( RegGetValueA( root, pKey, pValue, RRF_RT_REG_SZ, &type, pOut, &size ) != ERROR_SUCCESS )
		return false;
	return pOut[0] != 0;
}

// Looks for <library>\steamapps\common\GarrysMod\garrysmod\gameinfo.txt in
// the Steam install folder and every library listed in libraryfolders.vdf.
// pOutGameRoot receives ...\GarrysMod, pOutModDir ...\GarrysMod\garrysmod.
static bool FindGarrysMod( char *pOutGameRoot, int rootSize, char *pOutModDir, int modSize )
{
	char szSteam[MAX_PATH] = { 0 };
	if ( !ReadRegString( HKEY_CURRENT_USER, "Software\\Valve\\Steam", "SteamPath", szSteam, sizeof( szSteam ) ) &&
		 !ReadRegString( HKEY_LOCAL_MACHINE, "SOFTWARE\\WOW6432Node\\Valve\\Steam", "InstallPath", szSteam, sizeof( szSteam ) ) &&
		 !ReadRegString( HKEY_LOCAL_MACHINE, "SOFTWARE\\Valve\\Steam", "InstallPath", szSteam, sizeof( szSteam ) ) )
	{
		strcpy_s( szSteam, "C:\\Program Files (x86)\\Steam" );
	}

	const int kMaxLibs = 32;
	char libs[kMaxLibs][MAX_PATH];
	int numLibs = 0;
	strcpy_s( libs[numLibs++], szSteam );

	char szVdf[MAX_PATH];
	_snprintf_s( szVdf, _TRUNCATE, "%s\\steamapps\\libraryfolders.vdf", szSteam );
	FILE *fp = NULL;
	if ( fopen_s( &fp, szVdf, "rt" ) == 0 && fp )
	{
		char line[1024];
		while ( numLibs < kMaxLibs && fgets( line, sizeof( line ), fp ) )
		{
			// Expected form:   "path"    "D:\\SteamLibrary"
			char *pKey = strstr( line, "\"path\"" );
			if ( !pKey )
				continue;
			char *pStart = strchr( pKey + 6, '"' );
			if ( !pStart )
				continue;
			++pStart;
			char *pDst = libs[numLibs];
			int n = 0;
			for ( char *p = pStart; *p && *p != '"' && n < MAX_PATH - 1; ++p )
			{
				if ( p[0] == '\\' && p[1] == '\\' )
					++p;
				pDst[n++] = *p;
			}
			pDst[n] = 0;
			if ( n > 0 )
				++numLibs;
		}
		fclose( fp );
	}

	for ( int i = 0; i < numLibs; ++i )
	{
		for ( char *p = libs[i]; *p; ++p )
			if ( *p == '/' ) *p = '\\';

		char szRoot[MAX_PATH], szMod[MAX_PATH], szInfo[MAX_PATH];
		_snprintf_s( szRoot, _TRUNCATE, "%s\\steamapps\\common\\GarrysMod", libs[i] );
		_snprintf_s( szMod, _TRUNCATE, "%s\\garrysmod", szRoot );
		_snprintf_s( szInfo, _TRUNCATE, "%s\\gameinfo.txt", szMod );
		if ( FileExists( szInfo ) )
		{
			strncpy_s( pOutGameRoot, rootSize, szRoot, _TRUNCATE );
			strncpy_s( pOutModDir, modSize, szMod, _TRUNCATE );
			return true;
		}
	}
	return false;
}

static bool HasArg( int argc, char *argv[], const char *pName )
{
	for ( int i = 1; i < argc; ++i )
		if ( !_stricmp( argv[i], pName ) )
			return true;
	return false;
}

// True if a gameinfo.txt exists in the map's directory or any parent
// (Source's own LocateGameInfoFile() will then find the right game).
static bool MapLivesInsideGameDir( const char *pMapArg )
{
	char szPath[MAX_PATH];
	if ( !GetFullPathNameA( pMapArg, sizeof( szPath ), szPath, NULL ) )
		return false;
	for ( ;; )
	{
		char *pSlash = strrchr( szPath, '\\' );
		if ( !pSlash )
			return false;
		*pSlash = 0;
		char szInfo[MAX_PATH];
		_snprintf_s( szInfo, _TRUNCATE, "%s\\gameinfo.txt", szPath );
		if ( FileExists( szInfo ) )
			return true;
	}
}

// Uses the installed game's x64 tier0/vstdlib/filesystem_stdio instead of
// the embedded copies. Only enabled on request (-gamebin): ABI compatibility
// between GMod's runtime and this SDK build is not proven.
static bool CopyGameRuntimeDlls( const char *pGameRoot, const char *pRuntimeDir )
{
	static const char *s_Names[] = { "tier0.dll", "vstdlib.dll", "filesystem_stdio.dll" };
	char szSrc[3][MAX_PATH];
	for ( int i = 0; i < 3; ++i )
	{
		_snprintf_s( szSrc[i], _TRUNCATE, "%s\\bin\\win64\\%s", pGameRoot, s_Names[i] );
		if ( !FileExists( szSrc[i] ) || !IsX64Dll( szSrc[i] ) )
		{
			printf( "vvis launcher: %s missing or not x64, using embedded runtime.\n", szSrc[i] );
			return false;
		}
	}
	char szDst[MAX_PATH];
	_snprintf_s( szDst, _TRUNCATE, "%s\\tier0.dll", pRuntimeDir );
	if ( !CopyFileA( szSrc[0], szDst, FALSE ) ) return false;
	_snprintf_s( szDst, _TRUNCATE, "%s\\vstdlib.dll", pRuntimeDir );
	if ( !CopyFileA( szSrc[1], szDst, FALSE ) ) return false;
	_snprintf_s( szDst, _TRUNCATE, "%s\\bin\\x64\\filesystem_stdio.dll", pRuntimeDir );
	if ( !CopyFileA( szSrc[2], szDst, FALSE ) ) return false;
	return true;
}

// Extracts all embedded runtime DLLs into the private runtime directory
// (tier0.dll, vstdlib.dll, vvis_dll.dll) and into its bin\x64 subfolder
// (filesystem_stdio.dll, the relative path Sys_LoadModule()/PLATFORM_BIN_DIR
// expects; SDK_EXEC_DIR is pointed at this directory below).
static bool ExtractAllEmbeddedRuntimeDlls( char *pExeDir, int exeDirSize, const char *pGameRoot )
{
	GetRuntimeDirectory( pExeDir, exeDirSize );

	char szDest[MAX_PATH];

	_snprintf_s( szDest, _TRUNCATE, "%s\\tier0.dll", pExeDir );
	if ( !ExtractEmbeddedDll( IDR_TIER0_DLL, szDest ) )
		return false;

	_snprintf_s( szDest, _TRUNCATE, "%s\\vstdlib.dll", pExeDir );
	if ( !ExtractEmbeddedDll( IDR_VSTDLIB_DLL, szDest ) )
		return false;

	_snprintf_s( szDest, _TRUNCATE, "%s\\bin\\x64\\filesystem_stdio.dll", pExeDir );
	if ( !ExtractEmbeddedDll( IDR_FILESYSTEM_STDIO_DLL, szDest ) )
		return false;

	_snprintf_s( szDest, _TRUNCATE, "%s\\vvis_dll.dll", pExeDir );
	if ( !ExtractEmbeddedDll( IDR_VVIS_DLL, szDest ) )
		return false;

	if ( pGameRoot )
	{
		if ( CopyGameRuntimeDlls( pGameRoot, pExeDir ) )
		{
			printf( "vvis launcher: using runtime DLLs from %s\\bin\\win64\n", pGameRoot );
		}
		else
		{
			// Restore embedded copies (a partial copy may have happened).
			char szRedo[MAX_PATH];
			static const struct { int id; const char *rel; } s_Embedded[] = {
				{ IDR_TIER0_DLL, "tier0.dll" }, { IDR_VSTDLIB_DLL, "vstdlib.dll" }, { IDR_FILESYSTEM_STDIO_DLL, "bin\\x64\\filesystem_stdio.dll" } };
			for ( int i = 0; i < 3; ++i )
			{
				_snprintf_s( szRedo, _TRUNCATE, "%s\\%s", pExeDir, s_Embedded[i].rel );
				DeleteFileA( szRedo );
				if ( !ExtractEmbeddedDll( s_Embedded[i].id, szRedo ) )
					return false;
			}
		}
	}

	return true;
}

int main(int argc, char* argv[])
{
	// Auto-detect Garry's Mod when the user gave no -game/-vproject and the
	// map is not already inside a game folder containing gameinfo.txt.
	char szGameRoot[MAX_PATH] = { 0 };
	char szModDir[MAX_PATH] = { 0 };
	bool bHaveGame = FindGarrysMod( szGameRoot, sizeof( szGameRoot ), szModDir, sizeof( szModDir ) );
	bool bUseGameBin = HasArg( argc, argv, "-gamebin" );

	bool bInjectGame = argc >= 2 && !HasArg( argc, argv, "-game" ) && !HasArg( argc, argv, "-vproject" )
		&& argv[argc - 1][0] != '-' && !MapLivesInsideGameDir( argv[argc - 1] ) && bHaveGame;

	// Rebuild argv without -gamebin (unknown to vvis_dll) and with -game.
	// Windows reads a trailing \" (e.g. -game "C:\x\" "map.bsp") as an escaped
	// quote and merges the next argument into this one: <path>" <rest>.
	// Such an argument is split back into <path> and <rest>.
	char **newArgv = (char **)_alloca( sizeof( char * ) * ( argc * 2 + 3 ) );
	int newArgc = 0;
	newArgv[newArgc++] = argv[0];
	if ( bInjectGame )
	{
		printf( "vvis launcher: auto-detected game: %s\n", szModDir );
		newArgv[newArgc++] = (char *)"-game";
		newArgv[newArgc++] = szModDir;
	}
	for ( int i = 1; i < argc; ++i )
	{
		if ( !_stricmp( argv[i], "-gamebin" ) )
			continue;
		char *pQuote = strchr( argv[i], '"' );
		if ( pQuote )
		{
			char *pFirst = _strdup( argv[i] );
			pFirst[pQuote - argv[i]] = 0;
			char *pRest = pQuote + 1;
			while ( *pRest == ' ' )
				++pRest;
			char *pRestCopy = _strdup( pRest );
			size_t n = strlen( pRestCopy );
			while ( n > 0 && pRestCopy[n - 1] == '"' )
				pRestCopy[--n] = 0;
			newArgv[newArgc++] = pFirst;
			if ( *pRestCopy )
				newArgv[newArgc++] = pRestCopy;
			continue;
		}
		newArgv[newArgc++] = argv[i];
	}
	newArgv[newArgc] = NULL;

	char szExeDir[MAX_PATH];
	if ( !ExtractAllEmbeddedRuntimeDlls( szExeDir, sizeof( szExeDir ), ( bUseGameBin && bHaveGame ) ? szGameRoot : NULL ) )
	{
		printf( "vvis launcher error: failed to extract embedded runtime DLLs\n%s", GetLastErrorString() );
		return 1;
	}

	// FileSystem_GetExecutableDir() honours SDK_EXEC_DIR (and appends
	// bin\x64), so filesystem_stdio.dll is found in the runtime directory
	// regardless of where vvis_gpu.exe lives.
	SetEnvironmentVariableA( "SDK_EXEC_DIR", szExeDir );

	char szDllPath[MAX_PATH];
	_snprintf_s( szDllPath, _TRUNCATE, "%s\\vvis_dll.dll", szExeDir );

	// LOAD_WITH_ALTERED_SEARCH_PATH so vvis_dll.dll's own dependency
	// resolution (tier0.dll, vstdlib.dll) also looks next to szDllPath
	// first, matching the freshly extracted copies.
	HMODULE hModule = LoadLibraryExA( szDllPath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH );
	if ( !hModule )
	{
		printf( "vvis launcher error: can't load %s\n%s", szDllPath, GetLastErrorString() );
		return 1;
	}

	CreateInterfaceFn fn = (CreateInterfaceFn)GetProcAddress( hModule, "CreateInterface" );
	if ( !fn )
	{
		printf( "vvis launcher error: can't get CreateInterface from %s\n", szDllPath );
		FreeLibrary( hModule );
		return 2;
	}

	int retCode = 0;
	ILaunchableDLL *pDLL = (ILaunchableDLL*)fn( LAUNCHABLE_DLL_INTERFACE_VERSION, &retCode );
	if ( !pDLL )
	{
		printf( "vvis launcher error: can't get IVVisDLL interface from %s\n", szDllPath );
		FreeLibrary( hModule );
		return 3;
	}

	int result = pDLL->main( newArgc, newArgv );
	FreeLibrary( hModule );

	return result;
}
