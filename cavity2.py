#=======================================================================================================================
#                                       USER SETTINGS
#-----------------------------------------------------------------------------------------------------------------------

DlPrzX = 1.
DlPrzY = 1.

n = 20

dx = DlPrzX/n
dy = DlPrzY/n

niter = 200

x0, y0, dl = (0, 0, 0)

viscosity = 0.01

mom_relax = 0.7

outpath = "/home/wgryglas/fvcases/cavity/data"

#=======================================================================================================================

from mesh import Mesh
from fvm1 import *
from field import *
from interpolacja import *
from scipy.sparse.linalg.isolve.iterative import bicgstab
import gmsh_tools
import time

t = time.clock()

#node_c, cells, bound = siatka_regularna_prost(n, dx, dy, x0, y0)
#mesh = Mesh(node_c, cells, bound)

mesh = Mesh(*gmsh_tools.load("/home/wgryglas/SMESH/triangle_cavity/gmesh/triangle_cavity.msh"))
# mesh = Mesh(*gmsh_tools.load("/home/wgryglas/SMESH/cavity/gmesh/cavity.msh"))

#mesh.show()

print ">>>>> Mesh build in", time.clock()-t
t = time.clock()

Ux = SurfField(mesh, Dirichlet)
Uy = SurfField(mesh, Dirichlet)
p = SurfField(mesh, Neuman)

Ux.setBoundaryCondition(Dirichlet(mesh, 0, 1.))


np.set_printoptions(precision=3)

einterp = EdgeField.interp

Mxd, Fxd = laplace(viscosity, Ux)
Myd, Fyd = laplace(viscosity, Uy)

edgeU = EdgeField.vector(einterp(Ux), einterp(Uy))
phi = edgeU.dot(mesh.normals)
gradP = cellGrad(p)

for i in range(niter):
    print "========== iter", i, "============="
    edgeU = EdgeField.vector(einterp(Ux), einterp(Uy))
    phi = edgeU.dot(mesh.normals)
    gradP = cellGrad(p)

    Mxc, Fxc = div(phi, Ux)
    Myc, Fyc = div(phi, Uy)

    momX_M = Mxc - Mxd
    momY_M = Myc - Myd
    momX_F = -(Fxc - Fxd) - gradP[:, 0]*mesh.cells_areas
    momY_F = -(Fyc - Fyd) - gradP[:, 1]*mesh.cells_areas

    #Initial residuals
    print "Initial continuity residual:", np.linalg.norm(edgeDiv(phi))
    print "Initial Ux residual:", np.linalg.norm(momX_M.dot(Ux.data) - momX_F)
    print "Initial Uy residual:", np.linalg.norm(momY_M.dot(Uy.data) - momY_F)


    momX_M.relax3(momX_F, Ux, mom_relax)
    momY_M.relax3(momY_F, Uy, mom_relax)

    Ux.setValues(bicgstab(A=momX_M.sparse, b=momX_F, x0=Ux.data, tol=1e-8)[0])
    Uy.setValues(bicgstab(A=momY_M.sparse, b=momY_F, x0=Uy.data, tol=1e-8)[0])

    A = np.array([np.array(momX_M.diag), np.array(momY_M.diag)]).T

    coeffFieldX = SurfField(mesh, Neuman)
    coeffFieldX.setValues(mesh.cells_areas/A[:, 0])
    coeffFieldY = SurfField(mesh, Neuman)
    coeffFieldY.setValues(mesh.cells_areas/A[:, 1])
    coeffEdge = EdgeField.vector(EdgeField.interp(coeffFieldX), EdgeField.interp(coeffFieldY))

    edgeU = EdgeField.vector(einterp(Ux), einterp(Uy))
    phi = edgeU.dot(mesh.normals)

    #Rhie-Chow velocity interpolation
    # UxStar = SurfField(mesh, data=-(momX_M.offdiagmul(Ux.data)+Fxc-Fxd)/momX_M.diag)
    # UxStar.copyBoundaryConditions(Ux)
    # UyStar = SurfField(mesh, data=-(momY_M.offdiagmul(Uy.data)+Fyc-Fyd)/momY_M.diag)
    # UyStar.copyBoundaryConditions(Uy)
    # edgeU = EdgeField.vector(einterp(UxStar), einterp(UyStar))
    # phi = edgeU.dot(mesh.normals)


    Mpd, Fpd = laplace(coeffEdge, p)
    Fpd = -Fpd + edgeDiv(phi)

    pressP = SurfField(mesh, Neuman)
    pressP.setValues(bicgstab(A=Mpd.sparse, b=Fpd, x0=p.data, tol=1e-8)[0])

    p.setValues(p.data + (1.-mom_relax)*pressP.data)

    gradPP = cellGrad(pressP)

    Ux.setValues(Ux.data - gradPP[:, 0] * mesh.cells_areas/A[:, 0])
    Uy.setValues(Uy.data - gradPP[:, 1] * mesh.cells_areas/A[:, 1])


    print ">>>>> Computed in", time.clock()-t
    t = time.clock()


# animate_contour_plot([Ux.data.reshape((n,n))])
# plt.title("Ux")
#uyRes,
# animate_contour_plot([Uy.data.reshape((n,n))])
# plt.title("Uy")
#
# animate_contour_plot([p.data.reshape((n,n))])
# plt.title("p")
#
# Umag = np.sqrt(np.multiply(Ux.data, Ux.data) + np.multiply(Uy.data, Uy.data))
# animate_contour_plot([inter(mesh.xy, mesh.cells, Umag).reshape((n+1,n+1))], skip=1, repeat=False, interval=75, diff=viscosity, dt=1, adj=0)
#
# from matplotlib.pyplot import quiver
# q = quiver(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], Ux[:], Uy[:])
# plt.title("magU")

# plt.show()

import vtk_tools
vtk_tools.write_cell_data(outpath+"_cell.vtk", mesh, {"p": p}, {"U": [Ux, Uy]})
vtk_tools.write_point_data(outpath+"_pnt.vtk", mesh, {"p": p}, {"U": [Ux, Uy]})
