//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: Automatic update of vvis_gpu.exe from GitHub.
//
//=============================================================================//

#pragma once

// At startup, checks whether github.com/Lumino-2-0/GMOD-Source-SDK-Tools-GPU has a tag
// newer than this exe (the tag's commit date is compared with the exe's link time). If so:
// downloads the release asset, replaces vvis_gpu.exe (the old one becomes
// vvis_gpu.exe.old, deleted on the next launch) and restarts the new version with the
// same command line.
//
// Returns true if the new version was run: the launcher must then exit with *pExitCode.
// Otherwise returns false (up to date, offline, -noupdate, error...) and startup continues.
// The status is passed to vvis_dll through the VVIS_GPU_UPDATE_STATUS environment
// variable (shown in the banner; a "!" prefix marks a warning).
//
// Disable with -noupdate or VVIS_GPU_NOUPDATE=1.
bool VvisUpdater_Run( int argc, char *argv[], int *pExitCode );
