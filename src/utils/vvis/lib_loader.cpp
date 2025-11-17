#include "lib_loader.h"
#include "resource.h"

#include <windows.h>
#include <string>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

// ====================================================================
// Create directories recursively (bin/ and bin/x64/)
// ====================================================================
static void EnsureDirectory(const std::string& path)
{
    try
    {
        fs::create_directories(path);
    }
    catch (...)
    {
        // fallback Win32
        CreateDirectoryA(path.c_str(), nullptr);
    }
}



bool CheckRequiredDLLs()
{
    bool ok = true;

    auto Exists = [](const char* path) -> bool {
        DWORD attrib = GetFileAttributesA(path);
        return (attrib != INVALID_FILE_ATTRIBUTES && !(attrib & FILE_ATTRIBUTE_DIRECTORY));
        };

    if (!Exists("bin/x64/filesystem_stdio.dll"))
    {
        printf("\n[WARNING] filesystem_stdio.dll is missing.\n");
        printf("[INFO] It should be extracted automatically unless '-NoCheckLib' was used.\n");
        printf("[INFO] If extraction failed, delete the 'bin/' folder and relaunch vvis_GPU.\n\n");
        // pas fatal → extraction peut toujours le créer
    }

    return ok;
}


// ====================================================================
// Extract binary resource to a file
// ====================================================================
static bool ExtractResourceToFile(int resID, const std::string& outPath)
{
    HRSRC hRes = FindResourceA(NULL, MAKEINTRESOURCEA(resID), RT_RCDATA);
    if (!hRes)
    {
        printf("[LOADER] ERROR: FindResource failed for ID=%d\n", resID);
        return false;
    }

    HGLOBAL hData = LoadResource(NULL, hRes);
    if (!hData)
    {
        printf("[LOADER] ERROR: LoadResource failed\n");
        return false;
    }

    DWORD size = SizeofResource(NULL, hRes);
    void* ptr = LockResource(hData);

    if (!ptr || size == 0)
    {
        printf("[LOADER] ERROR: LockResource returned NULL\n");
        return false;
    }

    HANDLE hFile = CreateFileA(
        outPath.c_str(),
        GENERIC_WRITE,
        0, NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE)
    {
        printf("[LOADER] ERROR: Unable to create file: %s\n", outPath.c_str());
        return false;
    }

    DWORD written = 0;
    BOOL ok = WriteFile(hFile, ptr, size, &written, NULL);
    CloseHandle(hFile);

    if (!ok || written != size)
    {
        printf("[LOADER] ERROR: WriteFile failed for %s\n", outPath.c_str());
        return false;
    }

    return true;
}

// ====================================================================
// ENSURE ONLY filesystem_stdio.dll — this is what we want
// ====================================================================
void EnsureFilesystemOnly(bool silent)
{
    if (!silent)
        printf("[LOADER] Checking filesystem_stdio.dll...\n");

    // Required folders
    EnsureDirectory("bin");
    EnsureDirectory("bin/x64");

    std::string outPath = "bin/x64/filesystem_stdio.dll";

    // Extract the DLL
    bool ok = ExtractResourceToFile(IDR_FSSTDIO, outPath);

    if (!ok)
    {
        printf("[LOADER] ERROR: Failed to extract filesystem_stdio.dll\n");
    }
    else if (!silent)
    {
        printf("[LOADER] filesystem_stdio.dll extracted successfully.\n");
    }
}

