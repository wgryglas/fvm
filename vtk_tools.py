
def write_header_pnts(file_ptr, mesh, title):
    from os import linesep
    file_ptr.write("# vtk DataFile Version 2.0"+linesep)
    file_ptr.write(title+linesep)
    file_ptr.write("ASCII"+linesep)
    file_ptr.write("DATASET UNSTRUCTURED_GRID"+linesep)
    npts = len(mesh.xy)
    file_ptr.write("POINTS "+str(npts)+" float"+linesep)
    for p in mesh.xy:
        file_ptr.write(str(p[0])+" "+str(p[1])+" 0.0"+linesep)



def write_cell_data(file_path, mesh, scalars={}, vectors={}, title="fvm2D"):
    from os import linesep

    with open(file_path, "w") as f:
        write_header_pnts(f, mesh, title)

        nbEdges = sum([len(boundary) for boundary in mesh.boundaries])
        tot_len = sum([len(c) for c in mesh.cells]) + 2*nbEdges
        ncells = len(mesh.cells) + nbEdges

        f.write(linesep)
        f.write("CELLS "+str(ncells)+" "+str(tot_len+ncells)+linesep)
        for c in mesh.cells:
            f.write(str(len(c))+" "+" ".join(map(str, c))+linesep)

        for boundary in mesh.boundaries_points:
            for e in boundary:
                f.write(str(len(e))+" "+" ".join(map(str,e))+linesep)


        f.write(linesep)
        f.write("CELL_TYPES "+str(ncells)+linesep)
        for c in mesh.cells:
            if len(c) == 3:
                f.write("5"+linesep)
            elif len(c) == 4:
                f.write("9"+linesep)
            else:
                f.write("7"+linesep)
        f.write(linesep.join(["3"]*nbEdges))


        f.write(linesep)
        f.write("CELL_DATA "+str(ncells)+linesep)
        for var in scalars:
            field = scalars[var]
            f.write("SCALARS "+var+" float 1"+linesep)
            f.write("LOOKUP_TABLE default"+linesep)
            f.write(linesep.join(map(str, field.data))+linesep)
            for b in field.boundaries:
                f.write(linesep.join(map(str, b.data))+linesep)
            f.write(linesep)

        for var in vectors:
            xField = vectors[var][0]
            yField = vectors[var][1]
            f.write("VECTORS "+var+" float"+linesep)
            for x, y in zip(xField.data, yField.data):
                f.write(" ".join(map(str, [x, y, 0.0]))+linesep)

            for xBs, yBs in zip(xField.boundaries, yField.boundaries):
                for x,y in zip(xBs, yBs):
                    f.write(" ".join(map(str, [x, y, 0.0]))+linesep)

            f.write(linesep)


def cell_data_to_point_data(mesh, cellField):
    import numpy as np
    from field import EdgeField
    pntData = np.zeros(len(mesh.xy), dtype=float)
    influence = np.zeros(len(mesh.xy), dtype=int)

    edgeData = EdgeField.interp(cellField)

    for d, k in zip(edgeData.data, mesh.list_kr):
        influence[k[:2]] += 1
        pntData[k[:2]] += d

    pntData /= influence

    #force boundary nodes to be interpolated only by boundary data:
    for k in mesh.list_kr:
        if k[3] == -1:
            pntData[k[:2]] = 0.

    influence = np.ones(len(mesh.xy), dtype=int)
    influence[mesh.list_kr[mesh.boundaryEdges, :2]] = 0
    for d, k in zip(edgeData.data, mesh.list_kr):
        if k[3] == -1:
            pntData[k[:2]] += d
            influence[k[:2]] += 1

    pntData /= influence

    return pntData


def write_point_data(file_path, mesh, scalars={}, vectors={}, title="fvm2D"):
    from os import linesep

    for var in scalars:
        scalars[var] = cell_data_to_point_data(mesh, scalars[var])

    for var in vectors:
        vectors[var] = [cell_data_to_point_data(mesh, vectors[var][0]), cell_data_to_point_data(mesh, vectors[var][1])]

    with open(file_path, "w") as f:

        write_header_pnts(f, mesh, title)

        ncells = len(mesh.cells)
        tot_len = sum([len(c) for c in mesh.cells])

        f.write(linesep)
        f.write("CELLS "+str(ncells)+" "+str(tot_len+ncells)+linesep)
        for c in mesh.cells:
            f.write(str(len(c))+" "+" ".join(map(str, c))+linesep)

        f.write(linesep)
        f.write("CELL_TYPES "+str(ncells)+linesep)
        for c in mesh.cells:
            if len(c) == 3:
                f.write("5"+linesep)
            elif len(c) == 4:
                f.write("9"+linesep)
            else:
                f.write("7"+linesep)

        f.write(linesep)
        f.write("POINT_DATA "+str(len(mesh.xy))+linesep)

        for var in scalars:
            f.write("SCALARS "+var+" float 1"+linesep)
            f.write("LOOKUP_TABLE default"+linesep)
            f.write(linesep.join(map(str, scalars[var]))+linesep)
            f.write(linesep)

        for var in vectors:
            f.write("VECTORS "+var+" float"+linesep)
            for x, y in zip(vectors[var][0], vectors[var][1]):
                f.write(" ".join(map(str, [x, y, 0.0]))+linesep)
            f.write(linesep)
