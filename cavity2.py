#=======================================================================================================================
#                                       USER SETTINGS
#-----------------------------------------------------------------------------------------------------------------------

DlPrzX = 1.
DlPrzY = 1.

n = 40

dx = DlPrzX/n
dy = DlPrzY/n

niter = 100

x0, y0, dl = (0, 0, 0)

viscosity = 0.001

mom_relax = 0.7
#=======================================================================================================================

from mesh import Mesh
from fvm1 import *
from field import *
from interpolacja import *
from scipy.sparse.linalg.isolve.iterative import bicgstab

node_c, cells, bound = siatka_regularna_prost(n, dx, dy, x0, y0)

mesh = Mesh(node_c, cells, bound)


Ux = SurfField(mesh, Dirichlet)
Uy = SurfField(mesh, Dirichlet)
p = SurfField(mesh, Neuman)

Ux.setBoundaryCondition(Dirichlet(mesh, 2, 1.))

np.set_printoptions(precision=3)

einterp = EdgeField.interp

Mxd, Fxd = laplace(viscosity, Ux)
Myd, Fyd = laplace(viscosity, Uy)


for i in range(niter):
    print "iter", i

    edgeU = EdgeField.vector(einterp(Ux), einterp(Uy))
    phi = edgeU.dot(mesh.normals)

    gradP = cellGrad(p)

    Mxc, Fxc = div(phi, Ux)
    Myc, Fyc = div(phi, Uy)

    momX_M = Mxc - Mxd
    momY_M = Myc - Myd
    momX_F = -(Fxc - Fxd) - gradP[:, 0]*mesh.cells_areas
    momY_F = -(Fyc - Fyd) - gradP[:, 1]*mesh.cells_areas

    print "initial continuity error:", np.linalg.norm(edgeDiv(phi))
    print "initial Ux residual:", np.linalg.norm(momX_M.dot(Ux.data) - momX_F)
    print "initial Uy residual:", np.linalg.norm(momY_M.dot(Uy.data) - momY_F)

    momX_M.relax3(momX_F, Ux, mom_relax)
    momY_M.relax3(momY_F, Uy, mom_relax)

    Ux.setValues(bicgstab(A=momX_M.sparse, b=momX_F, x0=Ux.data, tol=1e-8)[0])

    Uy.setValues(bicgstab(A=momY_M.sparse, b=momY_F, x0=Uy.data, tol=1e-8)[0])

    print "final Ux residual:", np.linalg.norm(momX_M.dot(Ux.data) - momX_F)
    print "final Uy residual:", np.linalg.norm(momY_M.dot(Uy.data) - momY_F)

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



animate_contour_plot([Ux.data.reshape((n,n))])
plt.title("Ux")

animate_contour_plot([Uy.data.reshape((n,n))])
plt.title("Uy")

animate_contour_plot([p.data.reshape((n,n))])
plt.title("p")

Umag = np.sqrt(np.multiply(Ux.data, Ux.data) + np.multiply(Uy.data, Uy.data))
animate_contour_plot([inter(mesh.xy, mesh.cells, Umag).reshape((n+1,n+1))], skip=1, repeat=False, interval=75, diff=viscosity, dt=1, adj=0)

from matplotlib.pyplot import quiver
q = quiver(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], Ux[:], Uy[:])
plt.title("magU")

plt.show()
