import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera

# constants
#G=6.67*(10**(-11))
G=1
h=1
# Ma = 1*(10**10)
# Mb = 1*(10**10)
Ma = 1
Mb = 1

# iterations
N = 1000

# initial Conditions
qa0 = np.array([-10, 0])
pa0 = np.array([0, 0.15])
qb0 = np.array([10, 0])
pb0 = np.array([0, -0.15])


def Explicit_Eular(qa0, pa0, qb0, pb0):
    # solution variables
    qa = np.zeros([2, N])
    pa = np.zeros([2, N])
    qb = np.zeros([2, N])
    pb = np.zeros([2, N])

    # substitute initial conditions
    qa[:, 0] = qa0[:]
    pa[:, 0] = pa0[:]
    qb[:, 0] = qb0[:]
    pb[:, 0] = pb0[:]

    for i in range(N-1):
        k = G/((((qb[0, i] - qa[0, i])**2) + ((qb[1, i] - qa[1, i])**2))**(3/2))
        r = qb[:,i] - qa[:,i]
        qa[:, i+1] = qa[:, i] + h*pa[:, i]/Ma
        pa[:, i+1] = pa[:, i] + h*Ma*Mb*k*r
        qb[:, i + 1] = qb[:, i] + h * pb[:, i] / Mb
        pb[:, i + 1] = pb[:, i] - h * Ma * Mb * k * r

    return qa, pa, qb, pb


def Symmetric_Eular_VT(qa0, pa0, qb0, pb0):
    # solution variables
    qa = np.zeros([2, N])
    pa = np.zeros([2, N])
    qb = np.zeros([2, N])
    pb = np.zeros([2, N])

    # substitute initial conditions
    qa[:, 0] = qa0[:]
    pa[:, 0] = pa0[:]
    qb[:, 0] = qb0[:]
    pb[:, 0] = pb0[:]

    for i in range(N-1):
        k = G/((((qb[0, i] - qa[0, i])**2) + ((qb[1, i] - qa[1, i])**2))**(3/2))
        r = qb[:,i] - qa[:,i]
        pa[:, i + 1] = pa[:, i] + h * Ma * Mb * k * r
        qa[:, i+1] = qa[:, i] + h*pa[:, i+1]/Ma
        pb[:, i + 1] = pb[:, i] - h * Ma * Mb * k * r
        qb[:, i + 1] = qb[:, i] + h * pb[:, i+1] / Mb

    return qa, pa, qb, pb



def Symmetric_Eular_TV(qa0, pa0, qb0, pb0):
    # solution variables
    qa = np.zeros([2, N])
    pa = np.zeros([2, N])
    qb = np.zeros([2, N])
    pb = np.zeros([2, N])

    # substitute initial conditions
    qa[:, 0] = qa0[:]
    pa[:, 0] = pa0[:]
    qb[:, 0] = qb0[:]
    pb[:, 0] = pb0[:]

    for i in range(N-1):
        qa[:, i + 1] = qa[:, i] + h * pa[:, i] / Ma
        qb[:, i + 1] = qb[:, i] + h * pb[:, i] / Mb
        k = G/((((qb[0, i+1] - qa[0, i+1])**2) + ((qb[1, i+1] - qa[1, i+1])**2))**(3/2))
        r = qb[:,i+1] - qa[:,i+1]
        pa[:, i+1] = pa[:, i] + h*Ma*Mb*k*r
        pb[:, i + 1] = pb[:, i] - h * Ma * Mb * k * r

    return qa, pa, qb, pb


def animation(qa, qb, name):
    fig = plt.figure()
    plt.title(name)
    camera = Camera(fig)
    for i in range(N):
        plt.plot([qa[0,i]], [qa[1,i]], 'ro')
        plt.plot([qb[0, i]], [qb[1, i]], 'ro')
        camera.snap()
    animation = camera.animate(interval=10, repeat=False)
    plt.show()

qa1, pa1, qb1, pb1 = Explicit_Eular(qa0, pa0, qb0, pb0)
qa2, pa2, qb2, pb2 = Symmetric_Eular_VT(qa0, pa0, qb0, pb0)
qa3, pa3, qb3, pb3 = Symmetric_Eular_TV(qa0, pa0, qb0, pb0)

#plt.title("Gravitational Two Body Problem")
fig1, a =plt.subplots(2, 2)
plt.title("Gravitational Two Body Problem")

a[0][0].set_title("Ax")
a[0][0].plot(qa1[0,:], pa1[0,:], label='Explicit Eular')
a[0][0].plot(qa2[0,:], pa2[0,:], label='Symmetric Eular VT')
a[0][0].plot(qa3[0,:], pa3[0,:], label='Symmetric Eular TV')
a[0][0].legend(loc='lower right')

a[0][1].set_title("Ay")
a[0][1].plot(qa1[1,:], pa1[1,:], label='Explicit Eular')
a[0][1].plot(qa2[1,:], pa2[1,:], label='Symmetric Eular VT')
a[0][1].plot(qa3[1,:], pa3[1,:], label='Symmetric Eular TV')
a[0][1].legend(loc='lower right')

a[1][0].set_title("Bx")
a[1][0].plot(qb1[0,:], pb1[0,:], label='Explicit Eular')
a[1][0].plot(qb2[0,:], pb2[0,:], label='Symmetric Eular VT')
a[1][0].plot(qb3[0,:], pb3[0,:], label='Symmetric Eular TV')
a[1][0].legend(loc='lower right')

a[1][1].set_title("By")
a[1][1].plot(qb1[1,:], pb1[1,:], label='Explicit Eular')
a[1][1].plot(qb2[1,:], pb2[1,:], label='Symmetric Eular VT')
a[1][1].plot(qb3[1,:], pb3[1,:], label='Symmetric Eular TV')
a[1][1].legend(loc='lower right')

fig1.text(0.5, 0.04, 'q', ha='center', va='center')
fig1.text(0.06, 0.5, 'p', ha='center', va='center', rotation='vertical')

plt.show()


animation(qa1, qb1, 'Explicit Eular')
animation(qa2, qb2, "Symmetric_Eular_VT")
animation(qa3, qb3, "Symmetric_Eular_TV")