//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: Automatic update of vvis_gpu.exe from GitHub (see vvis_updater.h).
//
// GitHub API used (unauthenticated, 60 requests/hour/IP):
//   GET /repos/<repo>/tags                 -> name + commit sha of each tag
//   GET /repos/<repo>/git/commits/<sha>    -> commit date (cached)
//   GET /repos/<repo>/releases/tags/<tag>  -> downloadable assets
// A normal launch makes a single request (/tags): the dates of commits already seen
// and the tags already checked are kept in %LOCALAPPDATA%\vvis_gpu\update_cache.txt.
//
// The exe's reference date is the TimeDateStamp of its PE header (UTC link time). A tag
// only counts as newer if it is more than one hour past that date; the downloaded exe
// must also have a newer link time itself, otherwise the tag is marked "already checked"
// and is not downloaded again.
//
//=============================================================================//

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "vvis_updater.h"

#pragma comment( lib, "winhttp.lib" )

#define UPDATE_REPO			"Lumino-2-0/GMOD-Source-SDK-Tools-GPU"
#define UPDATE_API_ROOT		"https://api.github.com/repos/" UPDATE_REPO
#define UPDATE_EXE_NAME		"vvis_gpu.exe"
#define UPDATE_TOLERANCE	3600		// seconds required between the exe's link time and the tag's commit
#define MAX_COMMIT_LOOKUPS	6			// max /git/commits requests per launch

// --- Status passed to vvis_dll -------------------------------------------------------

static void SetStatus( bool bWarn, const char *pFmt, ... )
{
	char buf[512];
	buf[0] = bWarn ? '!' : ' ';
	va_list args;
	va_start( args, pFmt );
	_vsnprintf_s( buf + 1, sizeof( buf ) - 1, _TRUNCATE, pFmt, args );
	va_end( args );
	SetEnvironmentVariableA( "VVIS_GPU_UPDATE_STATUS", bWarn ? buf : buf + 1 );
}

static void Say( const char *pFmt, ... )
{
	va_list args;
	va_start( args, pFmt );
	printf( "vvis_gpu update: " );
	vprintf( pFmt, args );
	va_end( args );
	fflush( stdout );
}

// --- HTTP ------------------------------------------------------------------------------

struct HttpBuf
{
	char *p;
	size_t n, cap;
};

static bool BufAppend( HttpBuf &b, const void *pData, size_t len, size_t maxSize )
{
	if ( b.n + len + 1 > maxSize )
		return false;
	if ( b.n + len + 1 > b.cap )
	{
		size_t cap = b.cap ? b.cap : 65536;
		while ( cap < b.n + len + 1 )
			cap *= 2;
		char *p = (char *)realloc( b.p, cap );
		if ( !p )
			return false;
		b.p = p;
		b.cap = cap;
	}
	memcpy( b.p + b.n, pData, len );
	b.n += len;
	b.p[b.n] = 0;
	return true;
}

// HTTPS GET, following redirects. The body goes to *pOut (null-terminated) or is written to hFile.
// Returns the HTTP status code, or 0 if the connection or the transfer fails.
static int HttpGet( const char *pUrl, HttpBuf *pOut, HANDLE hFile, DWORD timeoutMs, size_t maxSize )
{
	wchar_t wUrl[2048];
	if ( !MultiByteToWideChar( CP_UTF8, 0, pUrl, -1, wUrl, 2048 ) )
		return 0;
	wchar_t host[256], path[2048], extra[1024];
	URL_COMPONENTS uc;
	memset( &uc, 0, sizeof( uc ) );
	uc.dwStructSize = sizeof( uc );
	uc.lpszHostName = host;		uc.dwHostNameLength = 256;
	uc.lpszUrlPath = path;		uc.dwUrlPathLength = 2048;
	uc.lpszExtraInfo = extra;	uc.dwExtraInfoLength = 1024;
	if ( !WinHttpCrackUrl( wUrl, 0, 0, &uc ) )
		return 0;
	if ( wcscat_s( path, extra ) != 0 )
		return 0;

	int status = 0;
	HINTERNET hSession = WinHttpOpen( L"vvis_gpu-updater", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0 );
	HINTERNET hConnect = NULL, hRequest = NULL;
	if ( hSession )
	{
		WinHttpSetTimeouts( hSession, timeoutMs, timeoutMs, timeoutMs, timeoutMs );
		hConnect = WinHttpConnect( hSession, host, uc.nPort, 0 );
	}
	if ( hConnect )
		hRequest = WinHttpOpenRequest( hConnect, L"GET", path, NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
			uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0 );
	if ( hRequest )
	{
		const wchar_t *pHeaders = !_wcsicmp( host, L"api.github.com" ) ?
			L"Accept: application/vnd.github+json\r\nX-GitHub-Api-Version: 2022-11-28\r\n" : WINHTTP_NO_ADDITIONAL_HEADERS;
		if ( WinHttpSendRequest( hRequest, pHeaders, (DWORD)-1L, WINHTTP_NO_REQUEST_DATA, 0, 0, 0 ) &&
			 WinHttpReceiveResponse( hRequest, NULL ) )
		{
			DWORD code = 0, size = sizeof( code );
			WinHttpQueryHeaders( hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &code, &size, WINHTTP_NO_HEADER_INDEX );
			status = (int)code;
			size_t total = 0;
			char chunk[65536];
			for ( ;; )
			{
				DWORD avail = 0, rd = 0;
				if ( !WinHttpQueryDataAvailable( hRequest, &avail ) || avail == 0 )
					break;
				if ( avail > sizeof( chunk ) )
					avail = sizeof( chunk );
				if ( !WinHttpReadData( hRequest, chunk, avail, &rd ) || rd == 0 )
				{
					status = 0;
					break;
				}
				total += rd;
				bool ok = total <= maxSize;
				if ( ok && pOut )
					ok = BufAppend( *pOut, chunk, rd, maxSize );
				if ( ok && hFile )
				{
					DWORD wr = 0;
					ok = WriteFile( hFile, chunk, rd, &wr, NULL ) && wr == rd;
				}
				if ( !ok )
				{
					status = 0;
					break;
				}
			}
		}
	}
	if ( hRequest )	WinHttpCloseHandle( hRequest );
	if ( hConnect )	WinHttpCloseHandle( hConnect );
	if ( hSession )	WinHttpCloseHandle( hSession );
	return status;
}

// --- Minimal JSON (GitHub API responses) ----------------------------------------------

// Finds "key": "value" starting at p. Copies the value and returns the position right
// after it, or NULL if the key is not found.
static const char *JsonString( const char *p, const char *pKey, char *pOut, size_t outSize )
{
	char pattern[64];
	_snprintf_s( pattern, sizeof( pattern ), _TRUNCATE, "\"%s\"", pKey );
	const size_t patLen = strlen( pattern );
	while ( p && ( p = strstr( p, pattern ) ) != NULL )
	{
		const char *q = p + patLen;
		while ( *q == ' ' || *q == '\t' || *q == '\r' || *q == '\n' )
			q++;
		if ( *q != ':' )
		{
			p = q;
			continue;
		}
		q++;
		while ( *q == ' ' || *q == '\t' || *q == '\r' || *q == '\n' )
			q++;
		if ( *q != '"' )
		{
			p = q;
			continue;
		}
		q++;
		size_t n = 0;
		while ( *q && *q != '"' )
		{
			if ( *q == '\\' && q[1] )
				q++;
			if ( n + 1 < outSize )
				pOut[n++] = *q;
			q++;
		}
		pOut[n] = 0;
		return *q ? q + 1 : q;
	}
	return NULL;
}

static __int64 ParseIsoDate( const char *pDate )
{
	struct tm t;
	memset( &t, 0, sizeof( t ) );
	if ( sscanf_s( pDate, "%d-%d-%dT%d:%d:%d", &t.tm_year, &t.tm_mon, &t.tm_mday, &t.tm_hour, &t.tm_min, &t.tm_sec ) != 6 )
		return 0;
	t.tm_year -= 1900;
	t.tm_mon -= 1;
	__int64 v = _mkgmtime64( &t );
	return v > 0 ? v : 0;
}

static void FormatDate( __int64 t, char *pOut, size_t size )
{
	struct tm tmv;
	if ( _gmtime64_s( &tmv, &t ) == 0 )
		strftime( pOut, size, "%Y-%m-%d %H:%M UTC", &tmv );
	else
		_snprintf_s( pOut, size, _TRUNCATE, "?" );
}

// --- Files ---------------------------------------------------------------------------

// Link time (UTC) of an exe on disk; fails if it is not an x64 (AMD64) PE image.
static bool ReadPeTime( const char *pPath, __int64 *pTime )
{
	HANDLE hFile = CreateFileA( pPath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL );
	if ( hFile == INVALID_HANDLE_VALUE )
		return false;
	IMAGE_DOS_HEADER dos;
	IMAGE_NT_HEADERS64 nt;
	memset( &nt, 0, sizeof( nt ) );
	DWORD rd = 0;
	bool ok = ReadFile( hFile, &dos, sizeof( dos ), &rd, NULL ) && rd == sizeof( dos ) && dos.e_magic == IMAGE_DOS_SIGNATURE
		&& SetFilePointer( hFile, dos.e_lfanew, NULL, FILE_BEGIN ) != INVALID_SET_FILE_POINTER
		&& ReadFile( hFile, &nt, sizeof( nt ), &rd, NULL ) && rd == sizeof( nt )
		&& nt.Signature == IMAGE_NT_SIGNATURE && nt.FileHeader.Machine == IMAGE_FILE_MACHINE_AMD64;
	CloseHandle( hFile );
	if ( ok )
		*pTime = (__int64)nt.FileHeader.TimeDateStamp;
	return ok;
}

static __int64 SelfLinkTime( void )
{
	const unsigned char *pBase = (const unsigned char *)GetModuleHandleA( NULL );
	const IMAGE_DOS_HEADER *pDos = (const IMAGE_DOS_HEADER *)pBase;
	const IMAGE_NT_HEADERS *pNt = (const IMAGE_NT_HEADERS *)( pBase + pDos->e_lfanew );
	return (__int64)pNt->FileHeader.TimeDateStamp;
}

static void RemoveTree( const char *pDir )
{
	char pattern[MAX_PATH];
	_snprintf_s( pattern, _TRUNCATE, "%s\\*", pDir );
	WIN32_FIND_DATAA fd;
	HANDLE h = FindFirstFileA( pattern, &fd );
	if ( h != INVALID_HANDLE_VALUE )
	{
		do
		{
			if ( !strcmp( fd.cFileName, "." ) || !strcmp( fd.cFileName, ".." ) )
				continue;
			char sub[MAX_PATH];
			_snprintf_s( sub, _TRUNCATE, "%s\\%s", pDir, fd.cFileName );
			if ( fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY )
				RemoveTree( sub );
			else
			{
				SetFileAttributesA( sub, FILE_ATTRIBUTE_NORMAL );
				DeleteFileA( sub );
			}
		} while ( FindNextFileA( h, &fd ) );
		FindClose( h );
	}
	RemoveDirectoryA( pDir );
}

static bool FindFileRecursive( const char *pDir, const char *pName, char *pOut, int outSize, int depth )
{
	char path[MAX_PATH];
	_snprintf_s( path, _TRUNCATE, "%s\\%s", pDir, pName );
	DWORD attrs = GetFileAttributesA( path );
	if ( attrs != INVALID_FILE_ATTRIBUTES && !( attrs & FILE_ATTRIBUTE_DIRECTORY ) )
	{
		strncpy_s( pOut, outSize, path, _TRUNCATE );
		return true;
	}
	if ( depth <= 0 )
		return false;
	char pattern[MAX_PATH];
	_snprintf_s( pattern, _TRUNCATE, "%s\\*", pDir );
	WIN32_FIND_DATAA fd;
	HANDLE h = FindFirstFileA( pattern, &fd );
	if ( h == INVALID_HANDLE_VALUE )
		return false;
	bool found = false;
	do
	{
		if ( !( fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY ) || !strcmp( fd.cFileName, "." ) || !strcmp( fd.cFileName, ".." ) )
			continue;
		char sub[MAX_PATH];
		_snprintf_s( sub, _TRUNCATE, "%s\\%s", pDir, fd.cFileName );
		found = FindFileRecursive( sub, pName, pOut, outSize, depth - 1 );
	} while ( !found && FindNextFileA( h, &fd ) );
	FindClose( h );
	return found;
}

static bool RunHidden( char *pCmdLine, DWORD timeoutMs )
{
	STARTUPINFOA si;
	PROCESS_INFORMATION pi;
	memset( &si, 0, sizeof( si ) );
	si.cb = sizeof( si );
	si.dwFlags = STARTF_USESHOWWINDOW;
	si.wShowWindow = SW_HIDE;
	if ( !CreateProcessA( NULL, pCmdLine, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi ) )
		return false;
	DWORD code = 1;
	if ( WaitForSingleObject( pi.hProcess, timeoutMs ) == WAIT_OBJECT_0 )
		GetExitCodeProcess( pi.hProcess, &code );
	else
		TerminateProcess( pi.hProcess, 1 );
	CloseHandle( pi.hThread );
	CloseHandle( pi.hProcess );
	return code == 0;
}

// tar.exe (Windows 10 1803+) can read zip files; PowerShell Expand-Archive is the fallback.
static bool ExtractZip( const char *pZip, const char *pDir )
{
	char sysDir[MAX_PATH];
	GetSystemDirectoryA( sysDir, sizeof( sysDir ) );
	char cmd[4096];
	_snprintf_s( cmd, _TRUNCATE, "\"%s\\tar.exe\" -xf \"%s\" -C \"%s\"", sysDir, pZip, pDir );
	if ( RunHidden( cmd, 120000 ) )
		return true;
	_snprintf_s( cmd, _TRUNCATE, "\"%s\\WindowsPowerShell\\v1.0\\powershell.exe\" -NoProfile -NonInteractive -Command \"Expand-Archive -LiteralPath '%s' -DestinationPath '%s' -Force\"",
		sysDir, pZip, pDir );
	return RunHidden( cmd, 120000 );
}

// --- Cache -----------------------------------------------------------------------------

#define MAX_CACHE 256

struct UpdateCache
{
	char sha[MAX_CACHE][48];
	__int64 date[MAX_CACHE];
	int numDates;
	char checked[MAX_CACHE][160];	// "<tag> <sha>" already downloaded and found not to be newer
	int numChecked;
	char path[MAX_PATH];
};

static void LoadCache( UpdateCache &c )
{
	c.numDates = c.numChecked = 0;
	c.path[0] = 0;
	char base[MAX_PATH];
	DWORD len = GetEnvironmentVariableA( "LOCALAPPDATA", base, sizeof( base ) );
	if ( len == 0 || len >= sizeof( base ) )
		return;
	char dir[MAX_PATH];
	_snprintf_s( dir, _TRUNCATE, "%s\\vvis_gpu", base );
	CreateDirectoryA( dir, NULL );
	_snprintf_s( c.path, _TRUNCATE, "%s\\update_cache.txt", dir );
	FILE *fp = NULL;
	if ( fopen_s( &fp, c.path, "rt" ) != 0 || !fp )
		return;
	char line[512];
	while ( fgets( line, sizeof( line ), fp ) )
	{
		line[strcspn( line, "\r\n" )] = 0;
		char a[160], b[160];
		long long v = 0;
		if ( c.numDates < MAX_CACHE && sscanf_s( line, "date %47s %lld", a, (unsigned)sizeof( a ), &v ) == 2 )
		{
			strcpy_s( c.sha[c.numDates], a );
			c.date[c.numDates++] = v;
		}
		else if ( c.numChecked < MAX_CACHE && sscanf_s( line, "checked %79s %47s", a, (unsigned)sizeof( a ), b, (unsigned)sizeof( b ) ) == 2 )
		{
			_snprintf_s( c.checked[c.numChecked++], _TRUNCATE, "%s %s", a, b );
		}
	}
	fclose( fp );
}

static void SaveCache( const UpdateCache &c )
{
	if ( !c.path[0] )
		return;
	FILE *fp = NULL;
	if ( fopen_s( &fp, c.path, "wt" ) != 0 || !fp )
		return;
	for ( int i = 0; i < c.numDates; i++ )
		fprintf( fp, "date %s %lld\n", c.sha[i], (long long)c.date[i] );
	for ( int i = 0; i < c.numChecked; i++ )
		fprintf( fp, "checked %s\n", c.checked[i] );
	fclose( fp );
}

static __int64 CachedDate( const UpdateCache &c, const char *pSha )
{
	for ( int i = 0; i < c.numDates; i++ )
		if ( !strcmp( c.sha[i], pSha ) )
			return c.date[i];
	return 0;
}

static bool IsChecked( const UpdateCache &c, const char *pTag, const char *pSha )
{
	char key[160];
	_snprintf_s( key, _TRUNCATE, "%s %s", pTag, pSha );
	for ( int i = 0; i < c.numChecked; i++ )
		if ( !strcmp( c.checked[i], key ) )
			return true;
	return false;
}

static void MarkChecked( UpdateCache &c, const char *pTag, const char *pSha )
{
	if ( IsChecked( c, pTag, pSha ) || c.numChecked >= MAX_CACHE )
		return;
	_snprintf_s( c.checked[c.numChecked++], _TRUNCATE, "%s %s", pTag, pSha );
	SaveCache( c );
}

// --- Update ----------------------------------------------------------------------------

static bool HasArgI( int argc, char *argv[], const char *pName )
{
	for ( int i = 1; i < argc; ++i )
		if ( !_stricmp( argv[i], pName ) )
			return true;
	return false;
}

// Picks the asset to download: vvis_gpu.exe itself, otherwise the zip whose name best
// matches "vvis" / "gpu".
static bool PickAsset( const char *pJson, char *pUrl, size_t urlSize )
{
	const char *pAssets = strstr( pJson, "\"assets\"" );
	if ( !pAssets )
		return false;
	char url[1024], zipUrl[1024] = { 0 };
	int zipScore = -1;
	const char *p = pAssets;
	while ( ( p = JsonString( p, "browser_download_url", url, sizeof( url ) ) ) != NULL )
	{
		const char *pName = strrchr( url, '/' );
		pName = pName ? pName + 1 : url;
		if ( !_stricmp( pName, UPDATE_EXE_NAME ) )
		{
			strncpy_s( pUrl, urlSize, url, _TRUNCATE );
			return true;
		}
		const size_t len = strlen( pName );
		if ( len > 4 && !_stricmp( pName + len - 4, ".zip" ) )
		{
			char lower[512];
			strncpy_s( lower, pName, _TRUNCATE );
			_strlwr_s( lower );
			const int score = ( strstr( lower, "vvis" ) ? 2 : 0 ) + ( strstr( lower, "gpu" ) ? 1 : 0 );
			if ( score > zipScore )
			{
				zipScore = score;
				strcpy_s( zipUrl, url );
			}
		}
	}
	if ( zipScore < 0 )
		return false;
	strncpy_s( pUrl, urlSize, zipUrl, _TRUNCATE );
	return true;
}

static bool CheckAndUpdate( int *pExitCode )
{
	char selfPath[MAX_PATH];
	if ( !GetModuleFileNameA( NULL, selfPath, sizeof( selfPath ) ) )
		return false;
	char oldPath[MAX_PATH];
	_snprintf_s( oldPath, _TRUNCATE, "%s.old", selfPath );
	DeleteFileA( oldPath );

	__int64 selfTime = SelfLinkTime();
	char fake[32];
	if ( GetEnvironmentVariableA( "VVIS_GPU_UPDATE_FAKETIME", fake, sizeof( fake ) ) > 0 )
		selfTime = _atoi64( fake );		// only for testing the update mechanism

	// 1. Tags and the date of their commit.
	HttpBuf tags = { 0 };
	const int st = HttpGet( UPDATE_API_ROOT "/tags?per_page=100", &tags, NULL, 2500, 4 << 20 );
	if ( st != 200 || !tags.p )
	{
		free( tags.p );
		if ( st == 403 || st == 429 )
			SetStatus( false, "check skipped (GitHub rate limit)" );
		else
			SetStatus( false, "check skipped (GitHub unreachable)" );
		return false;
	}

	UpdateCache *pCache = (UpdateCache *)calloc( 1, sizeof( UpdateCache ) );
	if ( !pCache )
	{
		free( tags.p );
		return false;
	}
	UpdateCache &cache = *pCache;
	LoadCache( cache );

	char bestTag[128] = { 0 }, bestSha[48] = { 0 };
	__int64 bestDate = 0;
	int lookups = 0;
	bool cacheDirty = false;
	const char *p = tags.p;
	char name[128], sha[48];
	while ( ( p = JsonString( p, "name", name, sizeof( name ) ) ) != NULL )
	{
		p = JsonString( p, "sha", sha, sizeof( sha ) );
		if ( !p )
			break;
		__int64 date = CachedDate( cache, sha );
		if ( !date && lookups < MAX_COMMIT_LOOKUPS )
		{
			lookups++;
			char url[512];
			_snprintf_s( url, _TRUNCATE, UPDATE_API_ROOT "/git/commits/%s", sha );
			HttpBuf commit = { 0 };
			if ( HttpGet( url, &commit, NULL, 2500, 1 << 20 ) == 200 && commit.p )
			{
				const char *pCommitter = strstr( commit.p, "\"committer\"" );
				char iso[64];
				if ( pCommitter && JsonString( pCommitter, "date", iso, sizeof( iso ) ) )
					date = ParseIsoDate( iso );
				if ( date && cache.numDates < MAX_CACHE )
				{
					strcpy_s( cache.sha[cache.numDates], sha );
					cache.date[cache.numDates++] = date;
					cacheDirty = true;
				}
			}
			free( commit.p );
		}
		if ( date > bestDate )
		{
			bestDate = date;
			strcpy_s( bestTag, name );
			strcpy_s( bestSha, sha );
		}
	}
	free( tags.p );
	if ( cacheDirty )
		SaveCache( cache );

	char dateStr[64];
	FormatDate( bestDate, dateStr, sizeof( dateStr ) );
	if ( !bestTag[0] )
	{
		SetStatus( false, "no release tag found" );
		free( pCache );
		return false;
	}
	if ( bestDate <= selfTime + UPDATE_TOLERANCE || IsChecked( cache, bestTag, bestSha ) )
	{
		SetStatus( false, "up to date (latest tag %s, %s)", bestTag, dateStr );
		free( pCache );
		return false;
	}

	// 2. Asset of the tag's release.
	char url[1024];
	_snprintf_s( url, _TRUNCATE, UPDATE_API_ROOT "/releases/tags/%s", bestTag );
	HttpBuf rel = { 0 };
	char assetUrl[1024];
	const bool bAsset = HttpGet( url, &rel, NULL, 2500, 4 << 20 ) == 200 && rel.p && PickAsset( rel.p, assetUrl, sizeof( assetUrl ) );
	free( rel.p );
	if ( !bAsset )
	{
		SetStatus( true, "new tag %s (%s) has no downloadable release yet", bestTag, dateStr );
		free( pCache );
		return false;
	}

	// 3. Download and extract into %TEMP%\vvis_gpu_update.
	char tempDir[MAX_PATH], workDir[MAX_PATH], filePath[MAX_PATH];
	GetTempPathA( sizeof( tempDir ), tempDir );
	_snprintf_s( workDir, _TRUNCATE, "%svvis_gpu_update", tempDir );
	RemoveTree( workDir );
	CreateDirectoryA( workDir, NULL );
	const char *pAssetName = strrchr( assetUrl, '/' );
	pAssetName = pAssetName ? pAssetName + 1 : "asset.zip";
	_snprintf_s( filePath, _TRUNCATE, "%s\\%s", workDir, pAssetName );

	Say( "new version %s (%s) found, downloading %s...\n", bestTag, dateStr, pAssetName );
	HANDLE hFile = CreateFileA( filePath, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL );
	int dl = 0;
	if ( hFile != INVALID_HANDLE_VALUE )
	{
		dl = HttpGet( assetUrl, NULL, hFile, 20000, (size_t)256 << 20 );
		CloseHandle( hFile );
	}
	char newExe[MAX_PATH] = { 0 };
	bool bReady = dl == 200;
	if ( bReady )
	{
		const size_t n = strlen( pAssetName );
		if ( n > 4 && !_stricmp( pAssetName + n - 4, ".zip" ) )
		{
			char outDir[MAX_PATH];
			_snprintf_s( outDir, _TRUNCATE, "%s\\x", workDir );
			CreateDirectoryA( outDir, NULL );
			bReady = ExtractZip( filePath, outDir ) && FindFileRecursive( outDir, UPDATE_EXE_NAME, newExe, sizeof( newExe ), 3 );
		}
		else
		{
			strcpy_s( newExe, filePath );
		}
	}
	__int64 newTime = 0;
	if ( bReady )
		bReady = ReadPeTime( newExe, &newTime );
	if ( !bReady )
	{
		SetStatus( true, "download of %s failed, keeping the current version", bestTag );
		RemoveTree( workDir );
		free( pCache );
		return false;
	}
	if ( newTime <= selfTime )
	{
		// The tag is newer but its exe is not: nothing to install.
		MarkChecked( cache, bestTag, bestSha );
		SetStatus( false, "up to date (latest tag %s, %s)", bestTag, dateStr );
		RemoveTree( workDir );
		free( pCache );
		return false;
	}

	// 4. Replace: the running exe can be renamed, but not overwritten.
	if ( !MoveFileExA( selfPath, oldPath, MOVEFILE_REPLACE_EXISTING ) )
	{
		SetStatus( true, "%s available but %s is not writable: https://github.com/" UPDATE_REPO "/releases/tag/%s",
			bestTag, selfPath, bestTag );
		RemoveTree( workDir );
		free( pCache );
		return false;
	}
	if ( !CopyFileA( newExe, selfPath, FALSE ) )
	{
		MoveFileExA( oldPath, selfPath, MOVEFILE_REPLACE_EXISTING );
		SetStatus( true, "could not install %s, keeping the current version", bestTag );
		RemoveTree( workDir );
		free( pCache );
		return false;
	}
	RemoveTree( workDir );
	free( pCache );

	// 5. Restart the new version with the same command line, in the same console.
	Say( "installed %s, restarting...\n\n", bestTag );
	SetEnvironmentVariableA( "VVIS_GPU_UPDATED", bestTag );
	STARTUPINFOA si;
	PROCESS_INFORMATION pi;
	memset( &si, 0, sizeof( si ) );
	si.cb = sizeof( si );
	char *pCmd = _strdup( GetCommandLineA() );
	const BOOL bStarted = CreateProcessA( selfPath, pCmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi );
	free( pCmd );
	SetEnvironmentVariableA( "VVIS_GPU_UPDATED", NULL );
	if ( !bStarted )
	{
		SetStatus( true, "%s installed, it will be used next time", bestTag );
		return false;
	}
	WaitForSingleObject( pi.hProcess, INFINITE );
	DWORD code = 1;
	GetExitCodeProcess( pi.hProcess, &code );
	CloseHandle( pi.hThread );
	CloseHandle( pi.hProcess );
	*pExitCode = (int)code;
	return true;
}

bool VvisUpdater_Run( int argc, char *argv[], int *pExitCode )
{
	char env[128];
	if ( GetEnvironmentVariableA( "VVIS_GPU_UPDATED", env, sizeof( env ) ) > 0 )
	{
		SetEnvironmentVariableA( "VVIS_GPU_UPDATED", NULL );
		SetStatus( false, "updated to %s", env );
		return false;
	}
	if ( HasArgI( argc, argv, "-noupdate" ) || GetEnvironmentVariableA( "VVIS_GPU_NOUPDATE", env, sizeof( env ) ) > 0 )
	{
		SetStatus( false, "check disabled (-noupdate)" );
		return false;
	}

	// Several vvis_gpu started at once: only one of them checks.
	HANDLE hMutex = CreateMutexA( NULL, FALSE, "Local\\vvis_gpu_update" );
	if ( hMutex && WaitForSingleObject( hMutex, 0 ) == WAIT_TIMEOUT )
	{
		CloseHandle( hMutex );
		SetStatus( false, "check skipped (another vvis_gpu is checking)" );
		return false;
	}
	const bool bRan = CheckAndUpdate( pExitCode );
	if ( hMutex )
	{
		ReleaseMutex( hMutex );
		CloseHandle( hMutex );
	}
	return bRan;
}
