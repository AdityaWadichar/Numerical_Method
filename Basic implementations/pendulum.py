import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera

#constants
m=1
g=9.81
l=9.81
k=(g/l)**(1/2)

#step size
h=np.pi/16

# initial conditins
q0=3/2
p0=0

# iterations
n=50

def Explicit_Eular(k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = sol[0,i] + h*sol[1,i]
        sol[1,i+1] = sol[1,i] - h*(k**2)*sol[0,i]

    return sol


# def Implicit_Eular(m, k, h, q0, p0):
#     sol=np.zeros([2,n])
#     sol[:,0]=np.array([[q0], [p0]])[:,0]
#
#     for i in range(n-1):
#         sol[0,i+1] = (sol[0,i] + h*sol[1,i]/m)*(1/(1+(h**2)*k/m))
#         sol[1,i+1] = (sol[1,i] - h*k*sol[0,i])*(1/(1+(h**2)*k/m))
#
#     return sol

def Symmetric_Eular_VT(k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[1, i + 1] = sol[1, i] - h * (k**2) * sol[0, i]
        sol[0,i+1] = sol[0,i] + h*sol[1,i+1]

    return sol

def Symmetric_Eular_TV(k, h, q0, p0):
    sol=np.zeros([2,n])
    sol[:,0]=np.array([[q0], [p0]])[:,0]

    for i in range(n-1):
        sol[0,i+1] = sol[0,i] + h*sol[1,i]
        sol[1,i+1] = sol[1,i] - h*(k**2)*sol[0,i+1]

    return sol


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

ans1 = Explicit_Eular(k, h, q0, p0)
#ans2 = Implicit_Eular(m, k, h, q0, p0)
ans3 = Symmetric_Eular_VT(k, h, q0, p0)
ans4 = Symmetric_Eular_TV(k, h, q0, p0)
fig=plt.figure()
plt.title('Mathematical Pendulum')
plt.xlabel('q')
plt.ylabel('p')
plt.plot(ans1[0,:], ans1[1,:], label='Explicit Eular')
#plt.plot(ans2[0,:], ans2[1,:], label='Implicit Eular')
plt.plot(ans3[0,:], ans3[1,:], label='Symmetric Eular VT')
plt.plot(ans4[0,:], ans4[1,:], label='Symmetric Eular TV')
plt.legend()
plt.show()
fig.savefig('Mathematical Pendulum')
# animation(ans1, l, 'Explicit Eular')
# animation(ans3, l, 'Symmetric Eular VT')
# animation(ans4, l, 'Symmetric Eular TV')
