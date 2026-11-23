// vvis_ui.cpp
//
// Console output for vvis_gpu (see vvis_ui.h).

#include <windows.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

#include "cmdlib.h"
#include "tier0/dbg.h"
#include "Color.h"
#include "vvis_ui.h"

bool g_bVisDebug = false;

static bool s_bColor = false;	// 4-bit colors through the spew (console only)
static bool s_bVT = false;		// 24-bit VT sequences (Windows 10+ console)

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

void VisUI_Init( int argc, char **argv )
{
	bool bNoColor = false;
	for ( int i = 1; i < argc; i++ )
	{
		if ( !_stricmp( argv[i], "-nocolor" ) )
			bNoColor = true;
		else if ( !_stricmp( argv[i], "-debug" ) )
			g_bVisDebug = true;
	}
	char tmp[8];
	if ( GetEnvironmentVariableA( "NO_COLOR", tmp, sizeof( tmp ) ) > 0 )
		bNoColor = true;

	HANDLE h = GetStdHandle( STD_OUTPUT_HANDLE );
	DWORD mode = 0;
	const bool bConsole = h && h != INVALID_HANDLE_VALUE && GetConsoleMode( h, &mode );
	s_bColor = bConsole && !bNoColor;
	s_bVT = false;
	if ( s_bColor )
		s_bVT = ( mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING ) || SetConsoleMode( h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING );
	if ( !bNoColor && GetEnvironmentVariableA( "FORCE_COLOR", tmp, sizeof( tmp ) ) > 0 && strcmp( tmp, "0" ) )
		s_bColor = s_bVT = true;
}

static Color SpewColor( VisUIColor c )
{
	// For CmdLib_SpewOutputFunc: non-zero r/g/b = channel on, a = intensity.
	switch ( c )
	{
	case UI_CYAN:		return Color( 0, 255, 255, 255 );
	case UI_GREEN:		return Color( 0, 255, 0, 255 );
	case UI_YELLOW:		return Color( 255, 255, 0, 255 );
	case UI_MAGENTA:	return Color( 255, 0, 255, 255 );
	case UI_BLUE:		return Color( 0, 0, 255, 255 );
	case UI_WHITE:		return Color( 254, 254, 254, 255 );
	case UI_GREY:		return Color( 0, 0, 0, 255 );
	default:			return Color( 255, 255, 255, 255 );
	}
}

static void ColorText( VisUIColor color, const char *pText )
{
	if ( s_bColor && color != UI_DEFAULT )
	{
		Color c = SpewColor( color );
		ColorSpewMessage( SPEW_MESSAGE, &c, "%s", pText );
	}
	else
	{
		Msg( "%s", pText );
	}
}

void VisUI_Msg( VisUIColor color, const char *pFmt, ... )
{
	char buf[4096];
	va_list args;
	va_start( args, pFmt );
	_vsnprintf_s( buf, sizeof( buf ), _TRUNCATE, pFmt, args );
	va_end( args );
	ColorText( color, buf );
}

// Text sent to the .log only (the console gets its own VT-decorated version).
void VisUI_LogOnly( const char *pFmt, ... )
{
	char buf[1024];
	va_list args;
	va_start( args, pFmt );
	_vsnprintf_s( buf, sizeof( buf ), _TRUNCATE, pFmt, args );
	va_end( args );
	const bool bOld = g_bSuppressPrintfOutput;
	g_bSuppressPrintfOutput = true;
	Msg( "%s", buf );
	g_bSuppressPrintfOutput = bOld;
}

// --- VT 24 bits ------------------------------------------------------------------

struct RGB3 { int r, g, b; };

static void VTColor( char *pOut, size_t size, RGB3 c )
{
	_snprintf_s( pOut, size, _TRUNCATE, "\x1b[38;2;%d;%d;%dm", c.r, c.g, c.b );
}

// Cyan -> blue -> magenta gradient, t in [0,1].
static RGB3 Gradient( float t )
{
	static const RGB3 stops[3] = { { 0, 255, 200 }, { 60, 140, 255 }, { 235, 70, 255 } };
	if ( t < 0.0f ) t = 0.0f;
	if ( t > 1.0f ) t = 1.0f;
	const float f = t * 2.0f;
	const int i = f >= 1.0f ? 1 : 0;
	const float u = f - (float)i;
	RGB3 c;
	c.r = (int)( stops[i].r + ( stops[i + 1].r - stops[i].r ) * u );
	c.g = (int)( stops[i].g + ( stops[i + 1].g - stops[i].g ) * u );
	c.b = (int)( stops[i].b + ( stops[i + 1].b - stops[i].b ) * u );
	return c;
}

static void VTPrint( const char *pText )
{
	fflush( stdout );
	fputs( pText, stdout );
	fflush( stdout );
}

#define VT_RESET "\x1b[0m"
#define VT_DIM   "\x1b[38;2;110;110;130m"
#define VT_TEXT  "\x1b[38;2;220;220;235m"
#define VT_CYAN  "\x1b[38;2;0;230;255m"
#define VT_GREEN "\x1b[1;38;2;80;255;120m"
#define VT_WARN  "\x1b[38;2;255;210;60m"

// --- Banner ----------------------------------------------------------------------

struct Glyph { const char *rows[5]; };

static const Glyph s_GlyphV = { { "__     __", "\\ \\   / /", " \\ \\ / / ", "  \\ V /  ", "   \\_/   " } };
static const Glyph s_GlyphI = { { " ___ ", "|_ _|", " | | ", " | | ", "|___|" } };
static const Glyph s_GlyphS = { { " ____  ", "/ ___| ", "\\___ \\ ", " ___) |", "|____/ " } };
static const Glyph s_GlyphUnder = { { "       ", "       ", "       ", "       ", " _____ " } };
static const Glyph s_GlyphG = { { "  ____ ", " / ___|", "| |  _ ", "| |_| |", " \\____|" } };
static const Glyph s_GlyphP = { { " ____  ", "|  _ \\ ", "| |_) |", "|  __/ ", "|_|    " } };
static const Glyph s_GlyphU = { { " _   _ ", "| | | |", "| | | |", "| |_| |", " \\___/ " } };

static void GetUpdateStatus( char *pOut, int size, bool *pbWarn )
{
	pOut[0] = 0;
	*pbWarn = false;
	if ( GetEnvironmentVariableA( "VVIS_GPU_UPDATE_STATUS", pOut, size ) == 0 )
	{
		pOut[0] = 0;
		return;
	}
	// The launcher prefixes states worth flagging with "!" (update failed, etc.).
	if ( pOut[0] == '!' )
	{
		*pbWarn = true;
		memmove( pOut, pOut + 1, strlen( pOut ) );
	}
}

void VisUI_Banner( void )
{
	char szUpdate[512];
	bool bUpdateWarn = false;
	GetUpdateStatus( szUpdate, sizeof( szUpdate ), &bUpdateWarn );

	if ( !s_bVT )
	{
		VisUI_Msg( UI_CYAN, "Valve + Lumastor & IRON-KNIGHT Software - vvis_gpu.exe (%s)\n", __DATE__ );
		if ( szUpdate[0] )
			VisUI_Msg( bUpdateWarn ? UI_YELLOW : UI_GREY, "update: %s\n", szUpdate );
		return;
	}

	VisUI_LogOnly( "Valve + Lumastor & IRON-KNIGHT Software - vvis_gpu.exe (%s)\n", __DATE__ );
	if ( szUpdate[0] )
		VisUI_LogOnly( "update: %s\n", szUpdate );

	const Glyph *text[] = { &s_GlyphV, &s_GlyphV, &s_GlyphI, &s_GlyphS, &s_GlyphUnder, &s_GlyphG, &s_GlyphP, &s_GlyphU };
	const int numGlyphs = sizeof( text ) / sizeof( text[0] );
	char rows[5][128];
	int width = 0;
	for ( int y = 0; y < 5; y++ )
	{
		rows[y][0] = 0;
		for ( int g = 0; g < numGlyphs; g++ )
			strcat_s( rows[y], text[g]->rows[y] );
		width = max( width, (int)strlen( rows[y] ) );
	}

	char out[16384];
	char col[32];
	out[0] = 0;
	strcat_s( out, "\n" );
	for ( int y = 0; y < 5; y++ )
	{
		strcat_s( out, "   " );
		const int len = (int)strlen( rows[y] );
		for ( int x = 0; x < len; x++ )
		{
			const char ch[2] = { rows[y][x], 0 };
			if ( ch[0] != ' ' )
			{
				VTColor( col, sizeof( col ), Gradient( (float)( x + 3 * y ) / (float)( width + 12 ) ) );
				strcat_s( out, col );
			}
			strcat_s( out, ch );
		}
		strcat_s( out, VT_RESET "\n" );
	}

	// Gradient separator line.
	strcat_s( out, "   " );
	for ( int x = 0; x < width; x++ )
	{
		VTColor( col, sizeof( col ), Gradient( (float)x / (float)width ) );
		strcat_s( out, col );
		strcat_s( out, ( x & 1 ) ? "-" : "=" );
	}
	strcat_s( out, VT_RESET "\n" );

	char line[512];
	_snprintf_s( line, sizeof( line ), _TRUNCATE,
		"   " VT_CYAN ">>" VT_TEXT " GPU raycast PVS compiler " VT_DIM "::" VT_TEXT " CUDA / OpenCL " VT_DIM "::" VT_TEXT " build %s" VT_RESET "\n"
		"   " VT_CYAN ">>" VT_DIM " Valve + Lumastor-IronKnight Software" VT_RESET "\n", __DATE__ );
	strcat_s( out, line );
	if ( szUpdate[0] )
	{
		_snprintf_s( line, sizeof( line ), _TRUNCATE, "   " VT_CYAN ">>" "%s update: %s" VT_RESET "\n",
			bUpdateWarn ? VT_WARN : VT_DIM, szUpdate );
		strcat_s( out, line );
	}
	if ( g_bVisDebug )
		strcat_s( out, "   " VT_CYAN ">>" VT_WARN " debug output enabled" VT_RESET "\n" );
	strcat_s( out, "\n" );
	VTPrint( out );
}

void VisUI_Label( const char *pLabel )
{
	VisUI_Msg( UI_CYAN, "%-20s ", pLabel );
}

void VisUI_PvsFinal( double percent )
{
	if ( !s_bVT )
	{
		VisUI_Msg( UI_GREEN, "PVS final: %.2f%%.\n", percent );
		return;
	}
	VisUI_LogOnly( "PVS final: %.2f%%.\n", percent );

	// Gauge: share of the map that is visible (green) vs hidden (grey), 40 cells.
	const int cells = 40;
	int on = (int)( percent * cells / 100.0 + 0.5 );
	if ( percent > 0.0 && on == 0 )
		on = 1;
	if ( on > cells )
		on = cells;
	char out[4096], col[32];
	_snprintf_s( out, sizeof( out ), _TRUNCATE, VT_CYAN "PVS final: " VT_GREEN "%.2f%%." VT_RESET "   " VT_DIM "[", percent );
	for ( int k = 0; k < cells; k++ )
	{
		if ( k < on )
		{
			RGB3 c = { 60 + 160 * k / cells, 255 - 40 * k / cells, 120 };
			VTColor( col, sizeof( col ), c );
			strcat_s( out, col );
			strcat_s( out, "#" );
		}
		else
		{
			strcat_s( out, VT_DIM "." );
		}
	}
	strcat_s( out, VT_DIM "] visible" VT_RESET "\n" );
	VTPrint( out );
}

void VisUI_Footer( const char *pElapsed )
{
	if ( !s_bVT )
	{
		VisUI_Msg( UI_GREEN, "%s elapsed\n", pElapsed );
		return;
	}
	VisUI_LogOnly( "%s elapsed\n", pElapsed );

	char head[256];
	_snprintf_s( head, sizeof( head ), _TRUNCATE, "==[ done : %s elapsed ]", pElapsed );
	const int width = 58;
	char out[8192], col[32];
	out[0] = 0;
	strcat_s( out, "\n   " );
	const int headLen = (int)strlen( head );
	for ( int x = 0; x < max( width, headLen + 4 ); x++ )
	{
		const char ch[2] = { x < headLen ? head[x] : ( ( x & 1 ) ? '-' : '=' ), 0 };
		VTColor( col, sizeof( col ), Gradient( (float)x / (float)width ) );
		strcat_s( out, col );
		strcat_s( out, ch );
	}
	strcat_s( out, VT_RESET "\n\n" );
	VTPrint( out );
}
