import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera
import phiKepler

e=0.2
h=0.1
T=500

q0=np.array([[1-e], [0]])
p0=np.array([[0], [((1+e)/(1-e))**(1/2)]])
t=h*np.arange(T)
mu=1

def Explicit_Eular(q0, p0, t, mu):
    q = np.zeros([2, len(t)])
    p = np.zeros([2, len(t)])

    q[0][0] = q0[0][0]
    q[1][0] = q0[1][0]
    p[0][0] = p0[0][0]
    p[1][0] = p0[1][0]

    for i in range(len(t)-1):
        q[0][i+1] = q[0][i] + h*p[0][i]
        q[1][i+1] = q[1][i] + h*p[1][i]

        p[0][i+1] = p[0][i] - h*mu*q[0][i]/(((q[0][i])**2 + (q[1][i])**2)**(3/2))
        p[1][i+1] = p[1][i] - h*mu*q[1][i] / (((q[0][i]) ** 2 + (q[1][i]) ** 2) ** (3 / 2))

    H = np.zeros([len(t)])
    for i in range(len(t)):
        H[i] = ((p[0][i]) ** 2 + (p[1][i]) ** 2) / 2 - mu / (((q[0][i]) ** 2 + (q[1][i]) ** 2) ** (1 / 2))

    return q, p, H


def Sympletic_Eular_VT(q0, p0, t, mu):
    q = np.zeros([2, len(t)])
    p = np.zeros([2, len(t)])

    q[0][0] = q0[0][0]
    q[1][0] = q0[1][0]
    p[0][0] = p0[0][0]
    p[1][0] = p0[1][0]

    for i in range(len(t)-1):
        p[0][i + 1] = p[0][i] - h * mu * q[0][i] / (((q[0][i]) ** 2 + (q[1][i]) ** 2) ** (3 / 2))
        p[1][i + 1] = p[1][i] - h * mu * q[1][i] / (((q[0][i]) ** 2 + (q[1][i]) ** 2) ** (3 / 2))

        q[0][i+1] = q[0][i] + h*p[0][i+1]
        q[1][i+1] = q[1][i] + h*p[1][i+1]

    H = np.zeros([len(t)])
    for i in range(len(t)):
        H[i] = ((p[0][i]) ** 2 + (p[1][i]) ** 2) / 2 - mu / (((q[0][i]) ** 2 + (q[1][i]) ** 2) ** (1 / 2))

    return q, p, H


def Sympletic_Eular_TV(q0, p0, t, mu):
    q = np.zeros([2, len(t)])
    p = np.zeros([2, len(t)])

    q[0][0] = q0[0][0]
    q[1][0] = q0[1][0]
    p[0][0] = p0[0][0]
    p[1][0] = p0[1][0]

    for i in range(len(t)-1):
        q[0][i+1] = q[0][i] + h*p[0][i]
        q[1][i+1] = q[1][i] + h*p[1][i]

        p[0][i+1] = p[0][i] - h*mu*q[0][i+1]/(((q[0][i+1])**2 + (q[1][i+1])**2)**(3/2))
        p[1][i+1] = p[1][i] - h*mu*q[1][i+1] / (((q[0][i+1]) ** 2 + (q[1][i+1]) ** 2) ** (3 / 2))

    H = np.zeros([len(t)])
    for i in range(len(t)):
        H[i] = ((p[0][i]) ** 2 + (p[1][i]) ** 2) / 2 - mu / (((q[0][i]) ** 2 + (q[1][i]) ** 2) ** (1 / 2))

    return q, p, H


def stormer_verlet(q0, p0, t, mu):
    M=1
    q = np.zeros([2, len(t)])
    q1 = np.zeros([2, len(t)])
    p = np.zeros([2, len(t)])

    q[0][0] = q0[0][0]
    q[1][0] = q0[1][0]
    p[0][0] = p0[0][0]
    p[1][0] = p0[1][0]

    for i in range(len(t)-1):
        q1[0][i+1] = q[0][i] + h*p[0][i]/(2*M)
        q1[1][i+1] = q[1][i] + h*p[1][i]/(2*M)

        p[0][i+1] = p[0][i] - h*mu*q1[0][i+1]/(((q1[0][i+1])**2 + (q1[1][i+1])**2)**(3/2))
        p[1][i + 1] = p[1][i] - h * mu * q1[1][i + 1] / (((q1[0][i + 1]) ** 2 + (q1[1][i + 1]) ** 2) ** (3 / 2))

        q[0][i+1] = q1[0][i+1] + h*p[0][i+1]/(2*M)
        q[1][i + 1] = q1[1][i + 1] + h * p[1][i + 1] / (2 * M)

    H = np.zeros([len(t)])
    for i in range(len(t)):
        H[i] = ((p[0][i])**2 + (p[1][i])**2)/2 - mu/(((q[0][i]) ** 2 + (q[1][i]) ** 2) ** (1 / 2))

    return q, p, H

def animation(ans1, name):
    fig = plt.figure()
    plt.title(name)
    camera = Camera(fig)
    for i in range(np.shape(ans1)[1]):
        plt.plot([ans1[0,i]], [ans1[1,i]], 'ro')
        plt.plot([0], [0], 'ro')
        camera.snap()
    animation = camera.animate(interval=50 ,repeat = False)
    #animation.save('Kepler.mp4')
    animation.save('Kepler.gif')

    plt.show()
    plt.close()


q, p, H = phiKepler.phiKepler(q0, p0, t, mu)
q1, p1, H1 = Explicit_Eular(q0, p0, t, mu)
q2, p2, H2 = Sympletic_Eular_VT(q0, p0, t, mu)
q3, p3, H3 = Sympletic_Eular_TV(q0, p0, t, mu)
q4, p4, H4 = stormer_verlet(q0, p0, t, mu)

H0 = H[0]

fig=plt.figure()
plt.plot(t, H-H0, 'green', label='Newton_Rapson')
plt.plot(t, H1-H0, 'red', label='Explicit_Eular')
plt.plot(t, H2-H0, 'blue', label='Sympletic_Eular_VT')
plt.plot(t, H3-H0, 'orange', label='Sympletic_Eular_TV')
plt.plot(t, H4-H0, 'yellow', label='stormer_verlet')
plt.xlabel('t')
plt.ylabel('H - H(0)')
plt.title('Kepler problem: Difference in hamiltonion')
plt.legend()
plt.show()
fig.savefig('Kepler_hamiltonion')

plt.plot(t, abs(np.sum(p1+q1-p-q, axis=0)), 'red', label='Explicit_Eular')
plt.plot(t, abs(np.sum(p2+q2-p-q, axis=0)), 'blue', label='Sympletic_Eular_VT')
plt.plot(t, abs(np.sum(p3+q3-p-q, axis=0)), 'orange', label='Sympletic_Eular_TV')
plt.plot(t, abs(np.sum(p4+q4-p-q, axis=0)), 'black', label='stormer_verlet')
plt.legend()
plt.show()

animation(q, 'Keplar')

