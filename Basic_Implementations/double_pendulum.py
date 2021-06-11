import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera

#constants
m1 = 1
m2 = 1
l1 = 9.81
l2 = 9.81
g = 9.81

#initial condition
q10 = np.pi/6
q20 = np.pi/8
p10 = 0
p20 = 0

q0 = [q10, q20]
p0 = [p10, p20]

h = 0.1 #step size
N = 1000 #iterations

def positon(q1, q2):
    pos1 = np.array([l1*np.sin(q1), -l1*np.cos(q1)])
    pos2 = pos1 + np.array([l2*np.sin(q2), -l2*np.cos(q2)])
    #print(pos1)
    return pos1, pos2


def stormer_verlet(q0, p0, h, N):
    sol = np.zeros([4, N])
    qh = np.zeros([2, N])

    sol[0, 0] = q0[0]
    sol[0, 1] = q0[1]
    sol[0, 2] = p0[0]
    sol[0, 3] = p0[1]

    for i in range(N-1):
        pos1, pos2 = positon(sol[0, i], sol[1, i])
        r12 = (pos1[0]) ** 2 + (pos1[1]) ** 2
        r22 = (pos2[0]) ** 2 + (pos2[1]) ** 2

        qh[0, i+1] = sol[0, i] + (h/2)*sol[2, i]/(m1*r12)
        qh[1, i+1] = sol[1, i] + (h/2)*sol[3, i]/(m2*r22)

        sol[2, i+1] = sol[2, i] - h*(m1+m2)*g*l1*np.sin(qh[0, i+1])
        sol[3, i+1] = sol[3, i] - h*m2*g*l2*np.sin(qh[1, i+1])

        pos1, pos2 = positon(qh[0, i+1], qh[1, i+1])
        r12 = (pos1[0]) ** 2 + (pos1[1]) ** 2
        r22 = (pos2[0]) ** 2 + (pos2[1]) ** 2

        sol[0, i+1] = qh[0, i+1] + (h/2)*sol[2, i+1]/(m1*r12)
        sol[1, i+1] = qh[1, i+1] + (h/2)*sol[3, i+1]/(m2*r22)

    H = np.zeros([N])
    KE = np.zeros([N])
    PE = np.zeros([N])
    for i in range(N):
        pos1, pos2 = positon(sol[0, i], sol[1, i])
        PE[i] = m1*g*pos1[1] + m2*g*pos2[1]
        r12 = (pos1[0])**2 + (pos1[1])**2
        r22 = (pos2[0])**2 + (pos2[1])**2
        KE[i] = (1/(2*m1*r12))*((sol[2, i])**2) + (1/(2*m2*r22))*((sol[3, i])**2)
        H[i] = PE[i] + KE[i]
    return sol, H, PE, KE


sol, H, PE, KE = stormer_verlet(q0, p0, h, N)
t=np.arange(N)
print(H[0])
plt.plot(h*t, H, label="H")
plt.plot(h*t, PE, label="PE")
plt.plot(h*t, KE, label="KE")
plt.legend()
plt.show()
