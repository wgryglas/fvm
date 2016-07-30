from mesh import Mesh
from numpy.linalg import solve
from fvm1 import *
from field import *
from interpolacja import *

np.set_printoptions(precision=3)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Zmienne w czasie zapis macierzy zadkich jako "wektory" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

DlPrzX = 1.; DlPrzY = 1.

n = 50                                                    # ilosc podzialow

dx = DlPrzX/n
dy = DlPrzY/n

dt = 0.0001                                                 # CFL u*dt/dx <= 1
tp = 0
tk = 0.01

nt = int((tk - tp)/dt)

x0, y0, dl = (0, 0, 0)


import time

# wsp_wezl, cells, bounduary = siatka_regularna_prost(n, dx, dy, x0, y0)  czyli metoda siatka_regularna_prost zwroci to co Mesh potrzebuje czyli nodes, cells, boundary

node_c, cells, bound = siatka_regularna_prost(n, dx, dy, x0, y0)

start = time.clock()
mesh = Mesh(node_c, cells, bound)                         # 1. tworzy obiekt mesh klasy Mesh, 2. generujemy siatke dla tego obiektu funkcja siatka_reg...
print ">>>> Mesh generated in " , time.clock()-start
start = time.clock()


# tworzy obiekt klasy SurfField, pobierajacy obirkt mesh klasy Mesh ( na tej siatce ma tworzyc i przechowywac rozwiazanie (wartosci))
T = SurfField(mesh, Dirichlet)

#T.setBoundaryCondition(Neuman(mesh, 0, 0))               # zero odpowiada zerowej krawedzi pobiera obiekt klasy Dirichlet (wywoluje go i tworzy)
#T.setBoundaryCondition(Dirichlet(mesh, 0, TdirWB))

#T.setBoundaryCondition(Neuman(mesh, 1, 0))
# T.setBoundaryCondition(Dirichlet(mesh, 1, TdirWB))

#T.setBoundaryCondition(Neuman(mesh, 2, 0))
#T.setBoundaryCondition(Dirichlet(mesh, 2, 1.))

#T.setBoundaryCondition(Neuman(mesh, 3, 0))              # symetria na krawedzi 3 (4)
# T.setBoundaryCondition(Dirichlet(mesh, 3, TdirWB))


for i, point in enumerate(T.data):
    if i < (n**2)/2:
        T.data[i] = 1.

T.updateBoundaryValues()


M, F = laplace(1, T)       #sLaplace(T)         # ukladanie macierzy i wektora prawych stron laplace

I = fvMatrix.diagonal(mesh, mesh.cells_areas/dt)

M = I - M

print ">>>> Equations generated in " , time.clock()-start
start = time.clock()

Results = list()
Tn = T.data.reshape((n, n))
Results.append(Tn)

from scipy.sparse.linalg.isolve.iterative import bicgstab

for iter in range(nt):
    print 'time iteration:',iter

    T.setValues( bicgstab(A=M.sparse, b=F+T.data*mesh.cells_areas/dt, x0=T.data,  tol=1e-8)[0] )

    Tn = T.data.reshape((n, n))
    Results.append(Tn)

print ">>>> Solved in ", time.clock()-start

# Animate results:
import matplotlib.pyplot as plt

# X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
# fig = plt.figure()
# plt.axes().set_aspect('equal', 'datalim')
# ticks = np.linspace(0, 10, 11)
# cs = plt.contourf(X, Y, T.data.reshape((n,n)), )
# cbar = fig.colorbar(cs, ticks=ticks)
# cbar.ax.set_yticklabels(map(str, ticks))
anim=animate_contour_plot(Results, skip=1, repeat=False, interval=5, dataRange=[0, 1])
plt.show()
#draw_values_edges(mesh.xy, mesh.cells, mesh.list_kr, T, n, DlPrzX, DlPrzY, Tdir)
#draw_edges(mesh.xy, mesh.list_kr)