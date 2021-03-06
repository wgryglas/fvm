
def skip_after(token, stream):
    line = stream.readline().replace("\n", "")
    count = 0
    while line != token:
        line = stream.readline().replace("\n", "")

def load(fname):
    import re
    import numpy as np
    import string

    with open(fname, "r") as f:
        skip_after("$PhysicalNames", f)
        nnames = int(f.readline())
        names = []
        ids = []
        for i in range(nnames):
            pname = re.findall(r'([0-9]+)\s([0-9]+)\s\"(([aA-zZ]|[0-9])+)\"', f.readline())[0][:3]
            if pname[0] == '1' and pname[2] != "defaultBoundary":
                bid = int(pname[1])
                names.append((bid-1, pname[2]))
                ids.append(bid)

        skip_after("$Nodes", f)
        nnodes = int(f.readline())
        nodes = np.zeros((nnodes, 2), dtype=float)
        for i in range(nnodes):
            nodes[i, :] = map(float, f.readline().replace("\n", "").split()[1:3])

        skip_after("$Elements", f)
        nelements = int(f.readline())
        boundary = []
        internal = []

        for i in range(nelements):
            info = map(int, f.readline().replace("\n", "").split())
            if info[3] in ids:
                boundary.append([d-1 for d in info[-2:]])
                boundary[-1].append(info[3]-1)
            elif info[1] == 2 or info[1] == 3:
                internal.append([d-1 for d in info[-(info[1]+1):]])
            elif info[1] != 1:
                raise Exception("Not supported element type with type id= "+str(info[1]))

        boundaryGrouped = []
        for i in range(len(names)):
            bdef = []
            for b in boundary:
                if b[-1] == i:
                    bdef.append(b[:2])
            boundaryGrouped.append(bdef)


        print "==========================="
        print "Loaded", fname, "mesh"
        print "Nr of nodes:", len(nodes)
        print "Nr of cells:", len(internal)
        print "Boundaries:"
        for name in names:
            print "--->", name[1], "has id =", name[0]
        print "==========================="

        return nodes, np.array(internal), boundaryGrouped

if __name__ == "__main__":
    #path = "/home/wgryglas/SMESH/cavity/gmesh/cavity.msh"
    path = "/home/wgryglas/SMESH/trojkaty/gmesh/trojkaty.msh"

    nodes, cells, boundaries = load(path)
    from mesh import Mesh
    mesh = Mesh(nodes, cells, boundaries)

    mesh.show()


