import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera

#constants
m=1
g=9.81
l=9.81
k=(g/l)**(1/2)

#step size
h=0.1

# initial conditins
q0=3/2
p0=0

# iterations
n=500

def Explicit_Eular(k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = sol[0,i] + h*sol[1,i]
        sol[1,i+1] = sol[1,i] - h*(k**2)*sol[0,i]

    H = np.zeros([n])
    for i in range(n):
        H[i] = -(1/2)*m*g*l*np.cos(sol[0,i]) + (1/(2*m))*((sol[1, i])**2)

    return sol, H


# def Implicit_Eular(m, k, h, q0, p0):
#     sol=np.zeros([2,n])
#     sol[:,0]=np.array([[q0], [p0]])[:,0]
#
#     for i in range(n-1):
#         sol[0,i+1] = (sol[0,i] + h*sol[1,i]/m)*(1/(1+(h**2)*k/m))
#         sol[1,i+1] = (sol[1,i] - h*k*sol[0,i])*(1/(1+(h**2)*k/m))
#
#     return sol

def Symplectic_Eular_VT(k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[1, i + 1] = sol[1, i] - h * (k**2) * sol[0, i]
        sol[0,i+1] = sol[0,i] + h*sol[1,i+1]

    H = np.zeros([n])
    for i in range(n):
        H[i] = -(1 / 2) * m * g * l * np.cos(sol[0, i]) + (1 / (2 * m)) * ((sol[1, i]) ** 2)

    return sol, H

def Symplectic_Eular_TV(k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = sol[0,i] + h*sol[1,i]
        sol[1,i+1] = sol[1,i] - h*(k**2)*sol[0,i+1]

    H = np.zeros([n])
    for i in range(n):
        H[i] = -(1 / 2) * m * g * l * np.cos(sol[0, i]) + (1 / (2 * m)) * ((sol[1, i]) ** 2)

    return sol, H

def stormer_verlet(k, h, qo, p0):
    sol = np.zeros([2, n])
    sol[:, 0] = np.array([[q0], [p0]])[:, 0]
    q1 = np.zeros([n])

    for i in range(n-1):
        q1[i+1] = sol[0,i] + h*sol[1,i]/2
        sol[1, i + 1] = sol[1, i] - h * (k ** 2) * q1[i + 1]
        sol[0, i + 1] = q1[i+1] + h * sol[1, i+1]/2

    H = np.zeros([n])
    for i in range(n):
        #H[i] = -m * g * l * np.cos(sol[0, i]) + (1 / (2 * m)) * ((sol[1, i]) ** 2)
        H[i] = -m * g * l * np.cos(sol[0, i]) + (1/2)*m*((g*h*i*np.cos(sol[0,i]))**2)

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
plt.title('Mathematical Pendulum')
plt.xlabel('q')
plt.ylabel('p')
plt.plot(ans1[0,:], ans1[1,:], label='Explicit Eular')
#plt.plot(ans2[0,:], ans2[1,:], label='Implicit Eular')
plt.plot(ans3[0,:], ans3[1,:], label='Symplectic Eular VT')
plt.plot(ans4[0,:], ans4[1,:], label='Symplectic Eular TV')
plt.legend()
plt.show()
#fig.savefig('Mathematical Pendulum')
# animation(ans1, l, 'Explicit Eular')
# animation(ans3, l, 'Symplectic Eular VT')
# animation(ans4, l, 'Symplectic Eular TV')
t=np.arange(n)
plt.plot(h*t, H5-H5[0])
plt.show()