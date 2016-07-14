from mesh import Mesh
from numpy.linalg import solve
from fvm1 import *
from field import *
from interpolacja import *

DlPrzX = 1.; DlPrzY = 1.

n = 50                                                    # ilosc podzialow

dx = DlPrzX/n
dy = DlPrzY/n

dt = 0.0001                                                 # CFL u*dt/dx <= 1
tp = 0
tk = 0.5

nt = (tk - tp)/dt

x0, y0, dl = (0, 0, 0)


import time

# wsp_wezl, cells, bounduary = siatka_regularna_prost(n, dx, dy, x0, y0)  czyli metoda siatka_regularna_prost zwroci to co Mesh potrzebuje czyli nodes, cells, boundary

node_c, cells, bound = siatka_regularna_prost(n, dx, dy, x0, y0)

start = time.clock()
mesh = Mesh(node_c, cells, bound)                         # 1. tworzy obiekt mesh klasy Mesh, 2. generujemy siatke dla tego obiektu funkcja siatka_reg...
print ">>>> Mesh generated in " , time.clock()-start
start = time.clock()


T = SurfField(mesh)                                       # tworzy obiekt klasy SurfField, pobierajacy obirkt mesh klasy Mesh ( na tej siatce ma tworzyc i przechowywac rozwiazanie (wartosci))

#print mesh.cell_centers


Tdir = 1
TdirWB = 0

T.setBoundaryCondition(Neuman(mesh, 0, 0))               # zero odpowiada zerowej krawedzi pobiera obiekt klasy Dirichlet (wywoluje go i tworzy)
#T.setBoundaryCondition(Dirichlet(mesh, 0, TdirWB))

T.setBoundaryCondition(Neuman(mesh, 1, 0))
#T.setBoundaryCondition(Dirichlet(mesh, 1, TdirWB))

T.setBoundaryCondition(Neuman(mesh, 2, 0))
#T.setBoundaryCondition(Dirichlet(mesh, 2, TdirWB))

T.setBoundaryCondition(Neuman(mesh, 3, 0))              # symetria na krawedzi 3 (4)
#T.setBoundaryCondition(Dirichlet(mesh, 3, TdirWB))

# d = Dirichlet(mesh, 3, TdirWB)
# d[:] = 1                                               # domyslnie zainicjalizowana zerami (dziedziczy po EdgeField) tu zapisuje te krawedz wartoscia = 1
# T.setBoundaryCondition(d)


M, F = sLaplace(T, dt)                                        # ukladanie macierzy i wektora prawych stron laplace


# M = M * (-dt)
F = F * dt

np.set_printoptions(precision=3)

#print Fc

pkt1 = n/2 + n*n/2                       # pkt srodek
pkt2 = n/2 + n*5                         # srodek 4 wiersze od spodu
pkt3 = n/2 + n*(n-5)                     # srodek 4 wiersze od gory

#F[pkt1] += -300
#F[pkt2] += -200
#F[pkt3] += -200


from scipy.sparse import csr_matrix

# I = csr_matrix((mesh.n**2, mesh.n**2), dtype=float)#np.matrix(np.identity(n*n))
# I.setdiag([1.]*mesh.n**2)


#M = I - M

T.data[:] = 0                          # War. Pocz. # [:] do listy przypisze wartosc 0, samo = przypisze inny obiekt przypisuje wszedzie wartosc 0
Fconst = -F

for i, point in enumerate(T.data):
    if i < (n**2)/2:
        T.data[i] = 10


print ">>>> Equations generated in " , time.clock()-start
start = time.clock()

Results = list()
Tn = T.data.reshape((n, n))
Results.append(Tn)


from scipy.sparse.linalg.isolve.iterative import bicgstab

for iter in range(int(nt)):
    print 'time iteration:',iter

    F = Fconst + T.data
    # T.data = np.array(np.linalg.solve(M, F))
    T.data = bicgstab(A=M, b=F, x0=T.data)[0]

    T.data = T.data.reshape((M.shape[0], 1))

    Tn = T.data.reshape((n, n))
    Results.append(Tn)

print ">>>> Solved in ", time.clock()-start

# Animate results:
animate_contour_plot(Results, skip=10, repeat=False, interval=75, dataRange=[0, 10])

#draw_values_edges(mesh.xy, mesh.cells, mesh.list_kr, T, n, DlPrzX, DlPrzY, Tdir)
#draw_edges(mesh.xy, mesh.list_kr)