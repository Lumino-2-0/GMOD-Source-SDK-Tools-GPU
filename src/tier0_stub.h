#pragma once

// --- Internal minimal Tier0 stub ---
// This replaces ALL tier0.dll imports.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cstdarg>
#include <chrono>

// ------------------------------------------------------------
// Memory system replacement
// ------------------------------------------------------------
inline void* Tier0_Malloc(size_t sz) { return malloc(sz); }
inline void  Tier0_Free(void* p) { free(p); }
inline void* Tier0_Realloc(void* p, size_t sz) { return realloc(p, sz); }
inline void* Tier0_AllocAlign(size_t sz, size_t align)
{
	void* out = nullptr;
#if _WIN32
	out = _aligned_malloc(sz, align);
#else
	posix_memalign(&out, align, sz);
#endif
	return out;
}
inline void Tier0_FreeAlign(void* p)
{
#if _WIN32
	_aligned_free(p);
#else
	free(p);
#endif
}

// ------------------------------------------------------------
// Logging replacement
// ------------------------------------------------------------
inline void Tier0_Msg(const char* fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);
}

inline void Tier0_Warning(const char* fmt, ...)
{
	fprintf(stderr, "[WARNING] ");
	va_list args;
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);
}

inline void Tier0_Error(const char* fmt, ...)
{
	fprintf(stderr, "[ERROR] ");
	va_list args;
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);
	exit(1);
}

// ------------------------------------------------------------
// Time replacement (Plat_FloatTime())
// ------------------------------------------------------------
inline double Plat_FloatTime()
{
	using namespace std::chrono;
	static auto start = high_resolution_clock::now();
	auto now = high_resolution_clock::now();
	auto diff = duration_cast<duration<double>>(now - start);
	return diff.count();
}

