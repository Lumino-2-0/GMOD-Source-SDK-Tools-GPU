#pragma once

// --- Minimal vstdlib stub ---
// Replaces all Q_* functions + Random + CommandLine

#include <cstring>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdarg>
#include <chrono>

// -----------------------------------------
// String helpers
// -----------------------------------------
inline int Q_stricmp(const char* s1, const char* s2)
{
	return _stricmp(s1, s2);
}

inline int Q_strnicmp(const char* s1, const char* s2, int n)
{
	return _strnicmp(s1, s2, n);
}

inline char* Q_strncpy(char* dest, const char* src, int maxlen)
{
	if (maxlen <= 0) return dest;
	strncpy_s(dest, maxlen, src, _TRUNCATE);
	return dest;
}

inline void Q_strncat(char* dest, const char* src, int maxlen)
{
	strncat_s(dest, maxlen, src, _TRUNCATE);
}

// -----------------------------------------
// Memory helpers
// -----------------------------------------
inline void Q_memcpy(void* d, const void* s, int size)
{
	memcpy(d, s, size);
}

inline void Q_memset(void* d, int c, int size)
{
	memset(d, c, size);
}

// -----------------------------------------
// Random
// -----------------------------------------
inline void RandomSeed(unsigned int seed)
{
	srand(seed);
}

inline int RandomInt(int min, int max)
{
	return min + (rand() % (max - min + 1));
}

// -----------------------------------------
// FloatTime (duplicate of Tier0 version)
// -----------------------------------------
inline double vstdlib_FloatTime()
{
	return Plat_FloatTime();
}

