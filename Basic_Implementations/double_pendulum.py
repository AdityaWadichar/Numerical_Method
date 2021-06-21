import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera

#constants
m1 = 1
m2 = 1
l1 = 10
l2 = 10
g = 9.81

#initial condition
q10 = np.pi/6
q20 = np.pi/2
p10 = 10
p20 = 10
dq10 = 0
dq20 = 0

q0 = [q10, q20]
p0 = [p10, p20]
dq0 = [dq10, dq20]

h = 0.1 #step size
N = 500 #iterations

def positon(q1, q2):
    pos1 = np.array([l1*np.sin(q1), -l1*np.cos(q1)])
    pos2 = np.array([l1*np.sin(q1)+l2*np.sin(q2), -l1*np.cos(q1)-l2*np.cos(q2)])
    #print(pos1)
    return pos1, pos2


def stormer_verlet(q0, p0, h, N):
    sol = np.zeros([4, N])
    qh = np.zeros([2, N])

    sol[0, 0] = q0[0]
    sol[1, 0] = q0[1]
    sol[2, 0] = p0[0]
    sol[3, 0] = p0[1]

    # for i in range(N-1):
    #     pos1, pos2 = positon(sol[0, i], sol[1, i])
    #     r12 = (pos1[0]) ** 2 + (pos1[1]) ** 2
    #     r22 = (pos2[0]) ** 2 + (pos2[1]) ** 2
    #
    #     qh[0, i+1] = sol[0, i] + (h/2)*sol[2, i]/(m1*r12)
    #     qh[1, i+1] = sol[1, i] + (h/2)*sol[3, i]/(m2*r22)
    #
    #     sol[2, i+1] = sol[2, i] - h*(m1+m2)*g*l1*np.sin(qh[0, i+1])
    #     sol[3, i+1] = sol[3, i] - h*m2*g*l2*np.sin(qh[1, i+1])
    #
    #     pos1, pos2 = positon(qh[0, i+1], qh[1, i+1])
    #     r12 = (pos1[0]) ** 2 + (pos1[1]) ** 2
    #     r22 = (pos2[0]) ** 2 + (pos2[1]) ** 2
    #
    #     sol[0, i+1] = qh[0, i+1] + (h/2)*sol[2, i+1]/(m1*r12)
    #     sol[1, i+1] = qh[1, i+1] + (h/2)*sol[3, i+1]/(m2*r22)
    #     #print((h/2)*sol[3, i+1]/(m2*r22))

    # relative momentum
    for i in range(N-1):
        qh[0, i+1] = sol[0, i] + (h/2)*((sol[2, i])/(m1*l1**2) + (m2*sol[2, i])/((m1**2)*(l1**2)) + (sol[3, i]*np.cos(sol[0, i] - sol[1, i]))/(m1*l1*l2))
        qh[1, i+1] = sol[1, i] + (h/2)*(sol[3, i]/(m2*(l2**2)) + (sol[2, i])*(np.cos(sol[0, i] - sol[1, i]))/(m1*l1*l2))

        sol[2, i+1] = sol[2, i] - h*((m1+m2)*g*l1*np.sin(qh[0, i+1]) - (sol[2, i]*sol[3, i]*np.sin(sol[0, i] - sol[1, i]))/(m1*l1*l2))
        sol[3, i+1] = sol[3, i] - h*(m2*g*l2*np.sin(qh[1, i+1]) + (sol[2, i]*sol[3, i]*np.sin(sol[0, i] - sol[1, i]))/(m1*l1*l2))

        sol[0, i + 1] = qh[0, i+1] + (h / 2) * (
                    (sol[2, i]) / (m1 * l1 ** 2) + (m2 * sol[2, i]) / ((m1 ** 2) * (l1 ** 2)) + (
                        sol[3, i] * np.cos(qh[0, i+1] - qh[1, i+1])) / (m1 * l1 * l2))
        sol[1, i + 1] = qh[1, i+1] + (h / 2) * (
                    sol[3, i] / (m2 * (l2 ** 2)) + (sol[2, i]) * (np.cos(qh[0, i+1] - qh[1, i+1])) / (m1 * l1 * l2))

    H = np.zeros([N])
    KE = np.zeros([N])
    PE = np.zeros([N])
    POS1 = np.zeros([2, N])
    POS2 = np.zeros([2, N])
    # for i in range(N):
    #     pos1, pos2 = positon(sol[0, i], sol[1, i])
    #     POS1[:,i] = pos1[:]
    #     POS2[:,i] = pos2[:]
    #     PE[i] = m1*g*pos1[1] + m2*g*pos2[1]
    #     r12 = (pos1[0])**2 + (pos1[1])**2
    #     r22 = (pos2[0])**2 + (pos2[1])**2
    #     KE[i] = (1/(2*m1*r12))*((sol[2, i])**2) + (1/(2*m2*r22))*((sol[3, i])**2)
    #     H[i] = PE[i] + KE[i]

    for i in range(N):
        PE[i] = -m1*g*l1*np.cos(sol[0, i]) - m2*g*(l1*np.cos(sol[0, i]) + l2*np.cos(sol[1, i]))
        KE[i] = (sol[2, i]**2)/(2*m1*l1**2) + (m1*sol[2, i]**2)/((m1*l1)**2) + (sol[3, i]**2)/(m2*(l1)**2) + (sol[2, i]*sol[3, i]*np.cos([sol[0, i] - sol[1, i]]))/(m1*l1*l2)
        KE[i] = KE[i]/2
        H[i] = PE[i] + KE[i]
        pos1, pos2 = positon(sol[0, i], sol[1, i])
        POS1[:,i] = pos1[:]
        POS2[:,i] = pos2[:]
    return sol, H, PE, KE, POS1, POS2


def lang(q0, p0, h, N):
    sol = np.zeros([4, N])
    qh = np.zeros([2, N])

    sol[0, 0] = q0[0]
    sol[1, 0] = q0[1]
    sol[2, 0] = p0[0]
    sol[3, 0] = p0[1]

    for i in range(N-1):
        q1 = sol[0, i]
        q2 = sol[1, i]
        dq1 = sol[2, i]
        dq2 = sol[3, i]

        q1 = sol[0, i] + (h/2)*dq1  #qh1
        q2 = sol[1, i] + (h / 2) * dq2  #qh2

        ddq1 = m2 * g * l1 * (np.sin(q2) * np.cos(q1 - q2) - np.sin(q1)) - m1 * g * l1 * np.sin(q1) - m2 * l1 * (
            np.sin(q1 - q2)) * (l1 * dq1 ** 2 + l2 * dq2 ** 2)
        ddq1 = ddq1 / ((m1 + m2 * (np.sin(q1 - q2)) ** 2) * l1 ** 2)

        ddq2 = -m2 * g * l2 * np.sin(q2) - m2 * l1 * l2 * ddq1 * np.cos(q1 - q2) + m2 * l1 * l2 * (dq1 ** 2) * np.sin(
            q1 - q2)
        ddq2 = ddq2 / (m2 * l2 ** 2)

        sol[2, i+1] = dq1 + h*ddq1
        sol[3, i+1] = dq2 + h*ddq2

        dq1 = sol[2, i+1]
        dq2 = sol[3, i+1]

        sol[0, i+1] = q1 + (h/2)*dq1
        sol[1, i+1] = q2 + (h/2)*dq2



    H = np.zeros([N])
    KE = np.zeros([N])
    PE = np.zeros([N])
    POS1 = np.zeros([2, N])
    POS2 = np.zeros([2, N])

    for i in range(N):
        q1 = sol[0, i]
        q2 = sol[1, i]
        dq1 = sol[2, i]
        dq2 = sol[3, i]

        KE[i] = (1/2)*m1*((l1*dq1)**2) + (1*2)*m2*((l1*dq1)**2 + (l2*dq2)**2 + 2*l1*l2*dq1*dq2*np.cos(q1-q2))
        PE[i] = -m1*g*l1*np.cos(q1) - m2*g*(l1*np.cos(q1) + l2*np.cos(q2))
        H[i] = KE[i] + PE[i]
        pos1, pos2 = positon(sol[0, i], sol[1, i])
        POS1[:, i] = pos1[:]
        POS2[:, i] = pos2[:]
    return sol, H, PE, KE, POS1, POS2









def animation(POS1, POS2, title):
    fig=plt.figure()
    plt.title(title)
    plt.xlim(-1.1*(l1+l2), 1.1*(l1+l2))
    plt.ylim(-1.1 * (l1 + l2), 1.1 * (l1 + l2))
    camera=Camera(fig)
    for i in range(N):
        plt.plot([0, POS1[0, i]], [0, POS1[1, i]], 'r-')
        plt.plot([POS1[0, i]], [POS1[1, i]], 'ro')
        plt.plot([POS1[0, i], POS2[0, i]], [POS1[1, i], POS2[1, i]], 'r-')
        plt.plot([POS2[0, i]], [POS2[1, i]], 'ro')
        camera.snap()
    animate = camera.animate(repeat=False)
    plt.show()


sol, H, PE, KE, POS1, POS2 = lang(q0, dq0, h, N)
t=np.arange(N)
fig=plt.figure()
plt.plot(h*t, H, label="Total Energy")
plt.plot(h*t, PE, label="Potential Energy")
plt.plot(h*t, KE, label="Kinetic Energy")
plt.title("Energy of double pendlum by stormer verlet method")
plt.xlabel("t")
plt.ylabel("Energy")
plt.legend()
plt.show()
#fig.savefig("double_pend_energy")
#print(POS1[0])
animation(POS1, POS2, "Double Pendulum by Stormer Verlet")
#print(sol[1, 0:15])
print(sol)

