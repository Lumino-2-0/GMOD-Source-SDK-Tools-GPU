"""Resout des offsets RVA de vrad_dll.dll en symboles via dbghelp (PDB a cote de la DLL).
Usage : python addr2sym.py chemin\\vrad_dll.dll 0x119EB 0x13326 ...
"""
import ctypes, sys
from ctypes import wintypes

class SYMBOL_INFO(ctypes.Structure):
    _fields_ = [('SizeOfStruct', wintypes.ULONG), ('TypeIndex', wintypes.ULONG), ('Reserved', ctypes.c_ulonglong * 2),
                ('Index', wintypes.ULONG), ('Size', wintypes.ULONG), ('ModBase', ctypes.c_ulonglong), ('Flags', wintypes.ULONG),
                ('Value', ctypes.c_ulonglong), ('Address', ctypes.c_ulonglong), ('Register', wintypes.ULONG), ('Scope', wintypes.ULONG),
                ('Tag', wintypes.ULONG), ('NameLen', wintypes.ULONG), ('MaxNameLen', wintypes.ULONG), ('Name', ctypes.c_char * 512)]

class IMAGEHLP_LINE64(ctypes.Structure):
    _fields_ = [('SizeOfStruct', wintypes.DWORD), ('Key', ctypes.c_void_p), ('LineNumber', wintypes.DWORD),
                ('FileName', ctypes.c_char_p), ('Address', ctypes.c_ulonglong)]

def main():
    dll = sys.argv[1]
    dbg = ctypes.WinDLL('dbghelp.dll')
    dbg.SymLoadModuleEx.restype = ctypes.c_ulonglong
    dbg.SymLoadModuleEx.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulonglong, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    h = wintypes.HANDLE(0x1234)
    dbg.SymSetOptions(0x12)  # UNDNAME | LOAD_LINES
    dbg.SymInitialize(h, None, False)
    base = 0x180000000
    dbg.SymLoadModuleEx(h, None, dll.encode(), None, base, 0, None, 0)
    for a in sys.argv[2:]:
        addr = base + int(a, 16)
        s = SYMBOL_INFO(); s.SizeOfStruct = 88; s.MaxNameLen = 511
        disp = ctypes.c_ulonglong(0)
        name = '?'
        if dbg.SymFromAddr(h, ctypes.c_ulonglong(addr), ctypes.byref(disp), ctypes.byref(s)):
            name = f"{s.Name.decode()}+0x{disp.value:x}"
        ln = IMAGEHLP_LINE64(); ln.SizeOfStruct = ctypes.sizeof(IMAGEHLP_LINE64)
        d32 = wintypes.DWORD(0)
        loc = ''
        if dbg.SymGetLineFromAddr64(h, ctypes.c_ulonglong(addr), ctypes.byref(d32), ctypes.byref(ln)):
            loc = f"{ln.FileName.decode()}:{ln.LineNumber}"
        print(f"{a:>10s}  {name}  {loc}")

if __name__ == '__main__':
    main()
