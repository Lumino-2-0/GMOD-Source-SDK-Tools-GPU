"""Diagnostic PVS : cluster d'un point, visibilite A->B, chemin de portails.
Usage : python pvs_diag.py map.bsp x1 y1 z1 x2 y2 z2
"""
import struct, sys
from collections import deque

def lump(d, i):
    off, ln, ver, _ = struct.unpack_from('<iiii', d, 8 + 16 * i)
    return off, ln, ver

def load(path):
    d = open(path, 'rb').read()
    o, l, _ = lump(d, 1)
    planes = [struct.unpack_from('<ffff', d, o + k * 20) for k in range(l // 20)]
    o, l, _ = lump(d, 5)
    nodes = [struct.unpack_from('<iii', d, o + k * 32) for k in range(l // 32)]
    o, l, v = lump(d, 10)
    sz = 32 if v == 1 else 56
    leafs = []
    for k in range(l // sz):
        c, cl, _a = struct.unpack_from('<ihh', d, o + k * sz)
        mn = struct.unpack_from('<hhh', d, o + k * sz + 8)
        mx = struct.unpack_from('<hhh', d, o + k * sz + 14)
        leafs.append((c, cl, mn, mx))
    o, l, _ = lump(d, 4)
    vis = d[o:o + l]
    return planes, nodes, leafs, vis

def find_leaf(planes, nodes, p):
    n = 0
    while n >= 0:
        pl, c0, c1 = nodes[n]
        a, b, c, dist = planes[pl]
        n = c0 if a * p[0] + b * p[1] + c * p[2] - dist >= 0 else c1
    return -1 - n

def pvs(vis, cl):
    nc = struct.unpack_from('<i', vis, 0)[0]
    ofs = struct.unpack_from('<i', vis, 4 + cl * 8)[0]
    out = set(); c = 0; i = ofs
    while c < nc:
        b = vis[i]
        if b == 0:
            c += 8 * vis[i + 1]; i += 2; continue
        for k in range(8):
            if b & (1 << k) and c + k < nc:
                out.add(c + k)
        c += 8; i += 1
    return nc, out

def portals(prt):
    lines = open(prt).read().splitlines()
    np_ = int(lines[2]); res = []
    for ln in lines[3:3 + np_]:
        p = ln.split()
        nw, a, b = int(p[0]), int(p[1]), int(p[2])
        pts = ln[ln.index('('):].replace('(', '').replace(')', '').split()
        w = [tuple(map(float, pts[k * 3:k * 3 + 3])) for k in range(nw)]
        res.append((a, b, w))
    return res

def main():
    bsp = sys.argv[1]
    A = tuple(map(float, sys.argv[2:5])); B = tuple(map(float, sys.argv[5:8]))
    planes, nodes, leafs, vis = load(bsp)
    la, lb = find_leaf(planes, nodes, A), find_leaf(planes, nodes, B)
    ca, cb = leafs[la][1], leafs[lb][1]
    print(f"A leaf {la} cluster {ca} contents {leafs[la][0]} bbox {leafs[la][2]} -> {leafs[la][3]}")
    print(f"B leaf {lb} cluster {cb} contents {leafs[lb][0]} bbox {leafs[lb][2]} -> {leafs[lb][3]}")
    nc, s = pvs(vis, ca)
    print(f"clusters {nc}, PVS(A) = {len(s)} clusters, B visible depuis A : {cb in s}")
    # Bornes de chaque cluster
    cb_box = {}
    for c, cl, mn, mx in leafs:
        if cl < 0: continue
        if cl in cb_box:
            o = cb_box[cl]; cb_box[cl] = (tuple(map(min, o[0], mn)), tuple(map(max, o[1], mx)))
        else:
            cb_box[cl] = (mn, mx)
    print(f"bbox cluster A : {cb_box[ca]}")
    P = portals(bsp[:-4] + '.prt')
    adj = {}
    for i, (a, b, w) in enumerate(P):
        adj.setdefault(a, []).append((b, i)); adj.setdefault(b, []).append((a, i))
    print(f"portails du cluster A : {len(adj.get(ca, []))}")
    for nb, i in adj.get(ca, []):
        w = P[i][2]
        zs = [p[2] for p in w]
        print(f"  -> cluster {nb:5d} via portail {i}, z {min(zs):.0f}..{max(zs):.0f}, pts {[tuple(int(x) for x in p) for p in w]}")
    # Chemin le plus court restreint au PVS de A
    prev = {ca: None}; q = deque([ca])
    while q:
        c = q.popleft()
        if c == cb: break
        for nb, i in adj.get(c, []):
            if nb in s and nb not in prev:
                prev[nb] = (c, i); q.append(nb)
    if cb in prev:
        path = []; c = cb
        while prev[c]:
            pc, i = prev[c]; path.append((pc, c, i)); c = pc
        print("chemin (dans le PVS) :")
        for pc, c, i in reversed(path):
            w = P[i][2]
            ctr = tuple(int(sum(p[k] for p in w) / len(w)) for k in range(3))
            print(f"  {pc} -> {c}  portail {i} centre {ctr} bbox cluster {cb_box.get(c)}")
            print(f"      pts {[tuple(int(x) for x in p) for p in w]}")

if __name__ == '__main__':
    main()
