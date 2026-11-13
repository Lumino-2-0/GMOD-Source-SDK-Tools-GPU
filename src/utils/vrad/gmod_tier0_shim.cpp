//=============================================================================//
// Purpose: fournit les symboles tier0 du SDK 2013 absents du tier0.dll x64 de
// Garry's Mod (qui utilise un tier0 plus recent), afin que vrad_dll.dll puisse
// se lier au tier0 de GMod (obligatoire pour charger son vphysics.dll).
// Ce fichier n'inclut volontairement aucun en-tete tier0.
//=============================================================================//
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace
{
	char s_TelemetryDummy[1024];
	float s_SpewColor[4] = { 0, 0, 0, 0 };
	void *s_pSpewFunc = NULL;

	bool ShimHushAsserts()
	{
		return false;
	}

	int ShimThreadWaitForObjects( int nEvents, const HANDLE *pHandles, bool bWaitAll, unsigned timeout )
	{
		DWORD r = WaitForMultipleObjects( nEvents, pHandles, bWaitAll ? TRUE : FALSE, timeout );
		return (int)r;
	}

	// Layout de CThreadFastMutex (x64) : m_ownerID (8 octets) puis m_depth (int).
	void ShimFastMutexLock( volatile void *pThis, unsigned __int64 threadId, unsigned nSpinSleepTime )
	{
		volatile unsigned __int64 *pOwner = (volatile unsigned __int64 *)pThis;
		volatile int *pDepth = (volatile int *)( (char *)pThis + 8 );
		for ( unsigned spins = 0; ; ++spins )
		{
			if ( *pOwner == threadId ||
				 InterlockedCompareExchange64( (volatile LONG64 *)pOwner, (LONG64)threadId, 0 ) == 0 )
				break;
			if ( spins > 100 )
				Sleep( nSpinSleepTime );
			else
				YieldProcessor();
		}
		MemoryBarrier();
		++( *pDepth );
	}

	void ShimSpewOutputFunc( void *pFunc )
	{
		s_pSpewFunc = pFunc;
	}

	const void *ShimGetSpewOutputColor()
	{
		return s_SpewColor;
	}

	void *ShimCommandLine()
	{
		typedef void *( *Fn )();
		// tier0 de GMod exporte "CommandLine", celui du SDK (runtime embarque) "CommandLine_Tier0".
		static Fn fn = NULL;
		if ( !fn )
		{
			HMODULE hTier0 = GetModuleHandleA( "tier0.dll" );
			fn = (Fn)GetProcAddress( hTier0, "CommandLine" );
			if ( !fn )
				fn = (Fn)GetProcAddress( hTier0, "CommandLine_Tier0" );
		}
		return fn ? fn() : NULL;
	}
}

extern "C"
{
	void *shim_imp_g_Telemetry = s_TelemetryDummy;
	void *shim_imp_HushAsserts = (void *)&ShimHushAsserts;
	void *shim_imp_ThreadWaitForObjects = (void *)&ShimThreadWaitForObjects;
	void *shim_imp_FastMutexLock = (void *)&ShimFastMutexLock;
	void *shim_imp_SpewOutputFunc = (void *)&ShimSpewOutputFunc;
	void *shim_imp_GetSpewOutputColor = (void *)&ShimGetSpewOutputColor;
	void *shim_imp_CommandLine_Tier0 = (void *)&ShimCommandLine;
}

#pragma comment( linker, "/alternatename:__imp_g_Telemetry=shim_imp_g_Telemetry" )
#pragma comment( linker, "/alternatename:__imp_HushAsserts=shim_imp_HushAsserts" )
#pragma comment( linker, "/alternatename:__imp_ThreadWaitForObjects=shim_imp_ThreadWaitForObjects" )
#pragma comment( linker, "/alternatename:__imp_?Lock@CThreadFastMutex@@AECAX_KI@Z=shim_imp_FastMutexLock" )
#pragma comment( linker, "/alternatename:__imp_SpewOutputFunc=shim_imp_SpewOutputFunc" )
#pragma comment( linker, "/alternatename:__imp_GetSpewOutputColor=shim_imp_GetSpewOutputColor" )
#pragma comment( linker, "/alternatename:__imp_CommandLine_Tier0=shim_imp_CommandLine_Tier0" )
