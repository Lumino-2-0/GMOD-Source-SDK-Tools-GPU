"""Liste les clusters du PVS d'un point, tries par distance, avec leur bbox.
Usage : python pvs_list.py map.bsp x y z [distance_min]
"""
import sys, math
from pvs_diag import load, find_leaf, pvs

def main():
    planes, nodes, leafs, vis = load(sys.argv[1])
    A = tuple(map(float, sys.argv[2:5]))
    dmin = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    ca = leafs[find_leaf(planes, nodes, A)][1]
    _, s = pvs(vis, ca)
    box = {}
    for c, cl, mn, mx in leafs:
        if cl < 0: continue
        if cl in box:
            o = box[cl]; box[cl] = (tuple(map(min, o[0], mn)), tuple(map(max, o[1], mx)))
        else:
            box[cl] = (mn, mx)
    rows = []
    for cl in s:
        mn, mx = box[cl]
        ctr = [(mn[k] + mx[k]) / 2 for k in range(3)]
        rows.append((math.dist(ctr, A), cl, mn, mx))
    rows.sort()
    print(f"cluster {ca}, PVS {len(s)} clusters")
    for d, cl, mn, mx in rows:
        if d >= dmin:
            print(f"{d:8.0f}  cluster {cl:5d}  {mn} -> {mx}")

if __name__ == '__main__':
    main()
