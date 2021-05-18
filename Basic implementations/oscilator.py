import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera

#constants
m=1
k=1

#step size
h=np.pi/10

# initial conditins
q0=3/2
p0=0

# iterations
n=25

def Explicit_Eular(m, k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = sol[0,i] + h*sol[1,i]/m
        sol[1,i+1] = sol[1,i] - h*k*sol[0,i]

    return sol


def Implicit_Eular(m, k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = (sol[0,i] + h*sol[1,i]/m)*(1/(1+(h**2)*k/m))
        sol[1,i+1] = (sol[1,i] - h*k*sol[0,i])*(1/(1+(h**2)*k/m))

    return sol

def Symmetric_Eular_VT(m, k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = sol[0,i]*(1-(h**2)*k/m) + h*sol[1,i]/m
        sol[1,i+1] = sol[1,i] - h*k*sol[0,i]

    return sol

def Symmetric_Eular_TV(m, k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = sol[0,i] + h*sol[1,i]/m
        sol[1,i+1] = sol[1,i]*(1-(h**2)*k/m) - h*k*sol[0,i]

    return sol


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

ans1 = Explicit_Eular(m, k, h, q0, p0)
ans2 = Implicit_Eular(m, k, h, q0, p0)
ans3 = Symmetric_Eular_VT(m, k, h, q0, p0)
ans4 = Symmetric_Eular_TV(m, k, h, q0, p0)
fig=plt.figure()
plt.title('Simple Harmonic Oscillator')
plt.xlabel('q')
plt.ylabel('p')
plt.plot(ans1[0,:], ans1[1,:], label='Explicit Eular')
plt.plot(ans2[0,:], ans2[1,:], label='Implicit Eular')
plt.plot(ans3[0,:], ans3[1,:], label='Symmetric Eular VT')
plt.plot(ans4[0,:], ans4[1,:], label='Symmetric Eular TV')
plt.legend()
plt.show()
fig.savefig('Simple Harmonic Oscillator')
# animation(ans1, 'Explicit Eular')
# animation(ans2, 'Implicit Eular')
# animation(ans3, 'Symmetric Eular VT')
# animation(ans4, 'Symmetric Eular TV')
#animation2(ans3, ans4)