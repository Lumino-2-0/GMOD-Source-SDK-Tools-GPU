// vvis_ui.h
//
// Console output for vvis_gpu: colors, ASCII banner, step labels.
//
// Messages always go through the cmdlib spew (console + .log file). 4-bit colors
// (SetConsoleTextAttribute) never end up in the .log. The banner and the 24-bit color
// gauges (VT sequences) are only written when stdout is a real console: inside Hammer
// (redirected output) or with -nocolor / NO_COLOR, the output stays plain text.
// FORCE_COLOR=1 forces colors even when the output is redirected.

#pragma once

// -debug: GPU raycast details (device, passes, statistics, detailed PVS).
extern bool g_bVisDebug;

enum VisUIColor
{
	UI_DEFAULT,
	UI_CYAN,
	UI_GREEN,
	UI_YELLOW,
	UI_MAGENTA,
	UI_BLUE,
	UI_WHITE,
	UI_GREY,
};

// Call before any output: reads -nocolor / -debug and detects the console.
void VisUI_Init( int argc, char **argv );
// Banner (console) or a plain header line (redirected output).
void VisUI_Banner( void );
// Step label padded to 20 columns (like "BasePortalVis:"), followed by the 0...10 progress bar.
void VisUI_Label( const char *pLabel );
void VisUI_Msg( VisUIColor color, const char *pFmt, ... );
// "PVS final: X%." line and its visible/hidden gauge.
void VisUI_PvsFinal( double percent );
// Closing line.
void VisUI_Footer( const char *pElapsed );
// Text written to the .log only (complements what threads.cpp prints with printf).
void VisUI_LogOnly( const char *pFmt, ... );
