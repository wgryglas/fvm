from mesh import Mesh
from numpy.linalg import solve
from fvm1 import *
from field import *
from interpolacja import *

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Zmienne w czasie zapis macierzy zadkich jako "wektory" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

DlPrzX = 1.; DlPrzY = 1.

n = 40                                                    # ilosc podzialow

dx = DlPrzX/n
dy = DlPrzY/n

dt = 0.001                                                 # CFL u*dt/dx <= 1
tp = 0
tk = 0.1

nt = (tk - tp)/dt

x0, y0, dl = (0, 0, 0)

viscosity = 0.01

import time

# wsp_wezl, cells, bounduary = siatka_regularna_prost(n, dx, dy, x0, y0)  czyli metoda siatka_regularna_prost zwroci to co Mesh potrzebuje czyli nodes, cells, boundary

node_c, cells, bound = siatka_regularna_prost(n, dx, dy, x0, y0)

mesh = Mesh(node_c, cells, bound)                         # 1. tworzy obiekt mesh klasy Mesh, 2. generujemy siatke dla tego obiektu funkcja siatka_reg...


Ux = SurfField(mesh, Dirichlet)                                       # tworzy obiekt klasy SurfField, pobierajacy obirkt mesh klasy Mesh ( na tej siatce ma tworzyc i przechowywac rozwiazanie (wartosci))
Uy = SurfField(mesh, Dirichlet)
p = SurfField(mesh, Neuman)

Ux.setBoundaryCondition(Dirichlet(mesh, 2, 1.))


np.set_printoptions(precision=3)


from fvMatrix import fvMatrix
einterp = EdgeField.interp

Mxd, Fxd = laplace(viscosity, Ux)
Myd, Fyd = laplace(viscosity, Uy)

# momX_M.relax(0.7)
# momY_M.relax(0.7)


from scipy.sparse.linalg.isolve.iterative import bicgstab

edgeU = EdgeField.vector(einterp(Ux), einterp(Uy))  # pole wektorowe predkosci [Ux, Uy] wyinterpolowanych wartosci na krawedzie (ze sr komurek  EdgeField.interp)
phi = edgeU.dot(mesh.normals)  # phi = v n A  gdzie An tu rowne jest dl_krawedzi obruconej o 90 stopni


# Correct phi to keep proper mass fluxes
# P, internIndex = adjustPhi_eqSys(phi)
# Pp = P[:, internIndex]
# from scipy.sparse import csr_matrix
# Mphi = Pp.dot(Pp.T)
# Mphi = csr_matrix(Mphi)
#
# def adjustPhiFlux():
#     from scipy.sparse.linalg import cg
#     Fphi = P.dot(phi.data)
#     Lambda = cg(Mphi, Fphi)[0]
#     dPhi = - Pp.T.dot(Lambda)
#     phi.data[internIndex] += dPhi


#adjustPhiFlux()

gradP = grad(p)

for i in range(30):
    print "iter", i

    Mxc, Fxc = div(phi, Ux)  # ukladanie macierzy i wektora prawych stron, dostaje D i Rhs z div
    Myc, Fyc = div(phi, Uy)

    momX_M = Mxc - Mxd
    momY_M = Myc - Myd

    momX_F = -(Fxc - Fxd) - gradP[:, 0]*mesh.cells_areas
    momY_F = -(Fyc - Fyd) - gradP[:, 1]*mesh.cells_areas

    # momX_M = Mxd
    # momY_M = Myd
    #
    # momX_F = gradP[:, 0]*mesh.cells_areas - Fxd
    # momY_F = gradP[:, 1]*mesh.cells_areas - Fyd

    print "initial continuity error:", np.linalg.norm(edgeDiv(phi))
    print "initial Ux residual:", np.linalg.norm(momX_M.dot(Ux.data) - momX_F)
    print "initial Uy residual:", np.linalg.norm(momY_M.dot(Uy.data) - momY_F)


    momX_M.relax(0.7)
    momY_M.relax(0.7)

    xSol = bicgstab(A=momX_M.sparse, b=momX_F, x0=Ux.data)[0]
    ySol = bicgstab(A=momY_M.sparse, b=momY_F, x0=Uy.data)[0]

    Ux.setValues(xSol)
    Uy.setValues(ySol)

    # Ux.setValues(0.3*Ux.data + 0.7*xSol)
    # Uy.setValues(0.3*Uy.data + 0.7*ySol)

    A = np.array([np.array(momX_M.diag), np.array(momY_M.diag)]).T

    # Hx = - momX_M.offdiagmul(Ux.data)
    # Hy = - momY_M.offdiagmul(Uy.data)

    # Ux.setValues(Hx / A)
    # Uy.setValues(Hy / A)

    edgeU = EdgeField.vector(einterp(Ux), einterp(Uy))
    phi = edgeU.dot(mesh.normals)


    Mpd, Fpd = laplace( 1./A, p) #mesh.cells_areas[:,np.newaxis]
    Fpd = -Fpd + edgeDiv(phi)

    pressSol = bicgstab(A=Mpd.sparse, b=Fpd, x0=p.data)[0]

    # pressP = SurfField(mesh, Neuman)
    # pressP.setValues(pressSol)

    # ptmp = np.copy(p.data)

    p.setValues(p.data + 0.3*(pressSol-p.data) )

    gradP = grad(p)

    Ux.setValues( Ux.data - gradP[:, 0] /A[:,0] ) #* mesh.cells_areas
    Uy.setValues( Uy.data - gradP[:, 1] /A[:,1] ) #* mesh.cells_areas

    # p.setValues(0.7 * ptmp + 0.3 * p.data)

    # gradP = grad(p)

    edgeU = EdgeField.vector(einterp(Ux), einterp(Uy))
    phi = edgeU.dot(mesh.normals)



    #adjustPhiFlux()

    # Ux.data = tmpX
    # Uy.data = tmpY





# Results = list()
# Tn = T.data.reshape((n, n))
# Results.append(Tn)
#
#

# animate_contour_plot([Ux.data.reshape((n,n))])
# plt.title("Ux")
#
# animate_contour_plot([Uy.data.reshape((n,n))])
# plt.title("Uy")
#
# animate_contour_plot([p.data.reshape((n,n))])
# plt.title("p")

Umag = np.sqrt(np.multiply(Ux.data, Ux.data) + np.multiply(Uy.data, Uy.data))
animate_contour_plot([inter(mesh.xy, mesh.cells, Umag).reshape((n+1,n+1))], skip=1, repeat=False, interval=75, diff=viscosity, dt=dt, adj=0)

# from matplotlib.pyplot import quiver
# q = quiver(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], Ux[:], Uy[:])
# plt.title("magU")
plt.show()



# animate_contour_plot([Uy.data.reshape((n,n))], dataRange=[0,1])
# plt.show()
#
#
# for iter in range(int(nt)):
#     print 'time iteration:',iter
#
#     F = Fconst + T.data
#     # T.data = np.array(np.linalg.solve(M, F))
#     T.data = bicgstab(A=M.sparse, b=F, x0=T.data)[0]
#
#     T.data = T.data.reshape((len(F), 1))
#
#     Tn = T.data.reshape((n, n))
#     Results.append(Tn)
#
# # Animate results:
# #animate_contour_plot(Results, skip=10, repeat=False, interval=75, dataRange=[0, 10])
#
# gradT = grad(T)
#
# from interpolacja import inter
# from matplotlib.pyplot import quiver
# from matplotlib.pyplot import contourf
# from numpy import meshgrid
#
#
# animate_contour_plot([inter(mesh.xy, mesh.cells, T.data).reshape((n+1,n+1))], skip=10, repeat=False, interval=75, dataRange=[0, 10])
#
# q = quiver(mesh.cell_centers[:,0], mesh.cell_centers[:,1], gradT[:, 0], gradT[:, 1])
#
# plt.show()



#draw_values_edges(mesh.xy, mesh.cells, mesh.list_kr, T, n, DlPrzX, DlPrzY, Tdir)
#draw_edges(mesh.xy, mesh.list_kr)


# start = time.clock()
# print ">>>> Solved in ", time.clock()-start
