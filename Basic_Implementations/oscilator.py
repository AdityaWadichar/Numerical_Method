import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera

#constants
m=1
k=1

#step size
h=0.1

# initial conditins
q0=3/2
p0=0

# iterations
n=100

def Explicit_Eular(m, k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = sol[0,i] + h*sol[1,i]/m
        sol[1,i+1] = sol[1,i] - h*k*sol[0,i]

    H = np.zeros([n])
    for i in range(n):
        H[i] = (1 / 2) * k * (sol[0, i])**2 + (1 / (2 * m)) * ((sol[1, i]) ** 2)

    return sol, H


def Implicit_Eular(m, k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = (sol[0,i] + h*sol[1,i]/m)*(1/(1+(h**2)*k/m))
        sol[1,i+1] = (sol[1,i] - h*k*sol[0,i])*(1/(1+(h**2)*k/m))

    H = np.zeros([n])
    for i in range(n):
        H[i] = (1 / 2) * k * (sol[0, i]) ** 2 + (1 / (2 * m)) * ((sol[1, i]) ** 2)

    return sol, H

def Symplectic_Eular_VT(m, k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = sol[0,i]*(1-(h**2)*k/m) + h*sol[1,i]/m
        sol[1,i+1] = sol[1,i] - h*k*sol[0,i]

    H = np.zeros([n])
    for i in range(n):
        H[i] = (1 / 2) * k * (sol[0, i]) ** 2 + (1 / (2 * m)) * ((sol[1, i]) ** 2)

    return sol, H

def Symplectic_Eular_TV(m, k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = sol[0,i] + h*sol[1,i]/m
        sol[1,i+1] = sol[1,i]*(1-(h**2)*k/m) - h*k*sol[0,i]

    H = np.zeros([n])
    for i in range(n):
        H[i] = (1 / 2) * k * (sol[0, i]) ** 2 + (1 / (2 * m)) * ((sol[1, i]) ** 2)

    return sol, H

def Analytical(m, k, h, q0, p0):
    sol = np.zeros([2, n])
    sol[:, 0] = np.array([[q0], [p0]])[:, 0]
    w= (k/m)**(1/2)

    for i in range(n):
        sol[0, i] = (np.cos(w*h*i))*q0 + (1/w)*(np.sin(w*i*h))*p0
        sol[1, i] = -w*(np.sin(w*i*h))*q0 + (np.cos(w*i*h))*p0
    H = np.zeros([n])
    for i in range(n):
        H[i] = (1 / 2) * k * (sol[0, i]) ** 2 + (1 / (2 * m)) * ((sol[1, i]) ** 2)

    return sol, H

def Stormer_verlet(m, k, h, q0, p0):
    sol = np.zeros([2, n])
    sol[:, 0] = np.array([[q0], [p0]])[:, 0]
    w = (k / m) ** (1 / 2)

    for i in range(n-1):
        sol[0, i+1] = sol[0, i] + (h*sol[1, i])/(2*m)
        sol[1, i+1] = sol[1, i] - h*k*sol[0, i+1]
        sol[0, i+1] = sol[0, i+1] + (h*sol[1, i+1])/(2 * m)
    H = np.zeros([n])
    for i in range(n):
        H[i] = (1 / 2) * k * (sol[0, i]) ** 2 + (1 / (2 * m)) * ((sol[1, i]) ** 2)

    return sol, H


def animation(ans1, name):
    fig = plt.figure()
    plt.title(name)
    camera = Camera(fig)
    for i in range(np.shape(ans1)[1]):
        plt.plot([ans1[0,i]], [0], 'ro')
        camera.snap()
    animation = camera.animate(repeat = False)

    plt.show()
    plt.close()

def animation2(ans1, ans2):
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(np.shape(ans1)[1]):
        plt.plot([ans1[0,i]], [0], 'ro')
        plt.plot([ans2[0, i]], [-2], 'bo')
        camera.snap()
    animation = camera.animate(repeat = False)

    plt.show()
    plt.close()
ans, H = Analytical(m, k, h, q0, p0)
ans1, H1 = Explicit_Eular(m, k, h, q0, p0)
ans2, H2 = Implicit_Eular(m, k, h, q0, p0)
ans3, H3 = Symplectic_Eular_VT(m, k, h, q0, p0)
ans4, H4 = Symplectic_Eular_TV(m, k, h, q0, p0)
ans5, H5 = Stormer_verlet(m, k, h, q0, p0)
fig=plt.figure()
plt.title('Simple Harmonic Oscillator (Phase plot)')
plt.xlabel('q')
plt.ylabel('p')
plt.plot(ans[0,:], ans[1,:], label='Analytical Solution')
plt.plot(ans1[0,:], ans2[1,:],'--', label='Explicit Euler')
plt.plot(ans2[0,:], ans2[1,:],'--', label='Implicit Euler')
plt.plot(ans3[0,:], ans3[1,:],'--', label='Symplectic Eular VT')
plt.plot(ans4[0,:], ans4[1,:],'-.', label='Symplectic Eular TV')
plt.plot(ans5[0,:], ans5[1,:],'--', label='Stormer_verlet')
plt.plot(q0, p0, ".", label="Initial condition")
plt.legend(loc='lower right')
plt.show()
fig.savefig('Oscillator_combine_phase')
#fig.savefig('Oscillator_Symplectic_phase')



fig=plt.figure()
plt.title('Simple Harmonic Oscillator (Energy plot)')
t=np.arange(n)
plt.xlabel('t (time)')
plt.ylabel('Total Energy')
plt.plot(h*t, H, label='Analytical Solution')
plt.plot(h*t, H1,'--', label='Explicit Eular')
plt.plot(h*t, H2,'--', label='Implicit Eular')
plt.plot(h*t, H3,'--', label='Symplectic Eular VT')
plt.plot(h*t, H4,'-.', label='Symplectic Eular TV')
plt.plot(h*t, H5,'--', label='Stormer_verlet')
plt.legend(loc='best')
plt.show()
fig.savefig('Oscillator_combine_energy')
#fig.savefig('Oscillator_Symplectic_energy')