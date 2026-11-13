"""Statistiques des tailles de lightmap des faces d'un BSP (detecte les faces > 32 luxels).
Usage : python bsp_lmsize.py map.bsp
"""
import struct, sys

def main():
    d = open(sys.argv[1], 'rb').read()
    off, ln = struct.unpack_from('<ii', d, 8 + 16 * 7)
    n = ln // 56
    big = []
    mx = 0
    for k in range(n):
        b = off + k * 56
        dispinfo = struct.unpack_from('<h', d, b + 12)[0]
        lofs = struct.unpack_from('<i', d, b + 20)[0]
        sz = struct.unpack_from('<ii', d, b + 36)
        w, h = sz[0] + 1, sz[1] + 1
        mx = max(mx, w * h)
        if lofs >= 0 and (sz[0] > 32 or sz[1] > 32):
            big.append((k, w, h, dispinfo))
    print(f"{n} faces, taille max {mx} luxels, faces > 32 : {len(big)}")
    for k, w, h, di in sorted(big, key=lambda x: -x[1] * x[2])[:20]:
        print(f"  face {k}: {w}x{h} ({w*h}) disp {di}")

if __name__ == '__main__':
    main()
