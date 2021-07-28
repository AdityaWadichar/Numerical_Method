import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera

#constants
m=1
g=9.81
l=10
k=(g/l)**(1/2)

#step size
h=0.1

# initial conditins
q0=np.pi/4
p0=-100

# iterations
n=100

def Explicit_Eular(k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = sol[0,i] + h*sol[1,i]/(m*(l**2))
        sol[1,i+1] = sol[1,i] - h*m*g*l*sol[0,i]

    H = np.zeros([n])
    PE = np.zeros([n])
    KE = np.zeros([n])
    for i in range(n):
        PE[i] = -m * g * l * np.cos(sol[0, i])
        KE[i] = (1 / (2 * m * (l ** 2))) * ((sol[1, i]) ** 2)
        H[i] = PE[i] + KE[i]
        #H1[i] = -m * g * l * np.cos(sol[0, i]) + (1 / 2) * m * ((g * h * i * np.cos(sol[0, i])) ** 2)
    return sol, H



def Symplectic_Eular_VT(k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[1, i + 1] = sol[1, i] - h * m*g*l * sol[0, i]
        sol[0,i+1] = sol[0,i] + h*sol[1,i+1]/(m*(l**2))

    H = np.zeros([n])
    PE = np.zeros([n])
    KE = np.zeros([n])
    for i in range(n):
        PE[i] = -m * g * l * np.cos(sol[0, i])
        KE[i] = (1 / (2 * m * (l ** 2))) * ((sol[1, i]) ** 2)
        H[i] = PE[i] + KE[i]
        #H1[i] = -m * g * l * np.cos(sol[0, i]) + (1 / 2) * m * ((g * h * i * np.cos(sol[0, i])) ** 2)
    return sol, H

def Symplectic_Eular_TV(k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = sol[0,i] + h*sol[1,i]/(m*(l**2))
        sol[1,i+1] = sol[1,i] - h*m*g*l*sol[0,i+1]

    H = np.zeros([n])
    PE = np.zeros([n])
    KE = np.zeros([n])
    for i in range(n):
        PE[i] = -m * g * l * np.cos(sol[0, i])
        KE[i] = (1 / (2 * m * (l ** 2))) * ((sol[1, i]) ** 2)
        H[i] = PE[i] + KE[i]
        #H1[i] = -m * g * l * np.cos(sol[0, i]) + (1 / 2) * m * ((g * h * i * np.cos(sol[0, i])) ** 2)
    return sol, H


def stormer_verlet(k, h, q0, p0):
    sol = np.zeros([2, n])
    sol[:, 0] = np.array([[q0], [p0]])[:, 0]
    q1 = np.zeros([n])
    I = m*l**2

    for i in range(n-1):
        q1[i+1] = sol[0,i] + (h/2)*sol[1,i]/(m*(l**2))
        sol[1, i + 1] = sol[1, i] - h *m*g*l * np.sin(q1[i + 1])
        sol[0, i + 1] = q1[i+1] + (h/2) * sol[1, i+1]/(m*(l**2))

    H = np.zeros([n])
    PE = np.zeros([n])
    KE = np.zeros([n])
    for i in range(n):
        PE[i] = -m * g * l * np.cos(sol[0, i])
        KE[i] = (1 / (2 * m * (l ** 2))) * ((sol[1, i]) ** 2)
        H[i] = PE[i] + KE[i]
        #H1[i] = -m * g * l * np.cos(sol[0, i]) + (1 / 2) * m * ((g * h * i * np.cos(sol[0, i])) ** 2)
    return sol, H


def animation(ans1, l, name):
    fig = plt.figure()
    plt.xlim(-1.2*l, 1.2*l)
    plt.ylim(-1.2 * l, 1.2 * l)
    plt.title(name)
    camera = Camera(fig)
    for i in range(np.shape(ans1)[1]):

        plt.plot([0, l*np.sin(ans1[0,i])], [0, -l*np.cos(ans1[0,i])], 'r-')
        plt.plot([l*np.sin(ans1[0,i])], [-l*np.cos(ans1[0,i])], 'ro')

        # plt.plot([ans2[0, i]], [-2], 'go', label='Implicit Eular')
        # plt.plot([ans3[0, i]], [-4], 'bo', label='Symmetric Eular VT')
        # plt.plot([ans4[0, i]], [-6], 'o', 'yellow', label='Symmetric Eular TV')
        #if i==0: plt.legend()
        camera.snap()
    animation = camera.animate(repeat = False)

    plt.show()
    plt.close()

ans1, H1 = Explicit_Eular(k, h, q0, p0)
#ans2, H2 = Implicit_Eular(m, k, h, q0, p0)
ans3, H3 = Symplectic_Eular_VT(k, h, q0, p0)
ans4, H4 = Symplectic_Eular_TV(k, h, q0, p0)
ans5, H5 = stormer_verlet(k, h, q0, p0)
fig=plt.figure()
plt.title('Mathematical Pendulum (phase plot)')
plt.xlabel('q')
plt.ylabel('p')
plt.plot(ans1[0,:], ans1[1,:], label='Explicit Eular')
#plt.plot(ans2[0,:], ans2[1,:], label='Implicit Eular')
plt.plot(ans3[0,:], ans3[1,:], label='Symplectic Eular VT')
plt.plot(ans4[0,:], ans4[1,:], label='Symplectic Eular TV')
plt.plot(ans5[0,:], ans5[1,:], label='Stormer Verlet')
plt.legend(loc='lower right')
plt.show()
fig.savefig('Pendulum_phase')
#animation(ans5, l, 'stormer-verlet')
# animation(ans3, l, 'Symplectic Eular VT')
# animation(ans4, l, 'Symplectic Eular TV')


t=np.arange(n)
fig2=plt.figure()
plt.plot(h*t, H1, label="Explicit Eular")
plt.plot(h*t, H3, label="Symplectic Eular VT")
plt.plot(h*t, H4, label="Symplectic Eular TV")
plt.plot(h*t, H5, label="Stormer Verlet")
plt.title("Mathematical Pendulum (energy plot)")
#plt.plot(h*t, H51, label="H1")
plt.xlabel("t (time)")
plt.ylabel("Total Energy")
plt.legend()
plt.show()
fig2.savefig("Pendulum_Energy")