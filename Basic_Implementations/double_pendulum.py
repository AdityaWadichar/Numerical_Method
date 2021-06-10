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

h = 0.01 #step size
N = 100 #iterations

def stormer_verlet(q0, p0, h, N):
    sol = np.zeros([4, N])
    qh = np.zeros([2, N])

    sol[0, 0] = q0[0]
    sol[0, 1] = q0[1]
    sol[0, 2] = p0[0]
    sol[0, 3] = p0[1]

    for i in range(N-1):
        qh[0, i+1] = sol[0, i] + (h/2)*sol[2, i]/m1
        qh[1, i+1] = sol[1, i] + (h/2)*sol[3, i]/m2

        sol[2, i+1] = sol[2, i] + (m1+m2)*h*g*l1*np.sin(sol[0, i])
        sol[3, i+1] = sol[3, i] + m2*g*l2*np.sin(sol[1, i])

        sol[0, i+1] = qh[0, i+1] + (h/2)*sol[2, i+1]/m1
        sol[1, i+1] = qh[1, i+1] + (h/2)*sol[3, i+1]/m2

    H = np.zeros([N])
    for i in range(N):
        H[i] = ((sol[2, i])**2)/(2*m1) + ((sol[3, i])**2)/(2*m2) - m1*g*l1*np.cos(sol[0, i]) - m2*g*(l1*np.cos(sol[0, i]) + l2*np.cos(sol[1, i]))

    return sol, H


sol, H = stormer_verlet(q0, p0, h, N)
t=np.arange(N)
print(H[0])
plt.plot(h*t, H)
plt.show()
