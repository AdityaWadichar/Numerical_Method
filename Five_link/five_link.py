import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera


# constants
# masses in kg
m1 = 2
m2 = 2
m3 = 2
m4 = m2
m5 = m1
m=[m1, m2, m3, m4, m5]

#lengths in m
l1 = 1
l2 = 1
l3 = 1
l4 = l2
l5 = l1

g = 9.81
#T = 10
N = 1000
h = 0.02
O = [0, 0]


#initial conditions
q10 = np.pi/2 + np.pi/6
q20 = np.pi/2 + np.pi/3
q30 = np.pi/2 + np.pi/15
q40 = np.pi/3.5
q50 = np.pi/6.5

p10 = -10
p20 = -5
p30 = -5
p40 = 10
p50 = 10


q0 = [q10, q20, q30, q40, q50]
p0 = [p10, p20, p30, p40, p50]


def position(q, O):
    com = np.zeros([2, 5])
    link = np.zeros([2, 5])

    com[0][0] = (l1/2) * np.cos(q[0]) + O[0]
    com[1][0] = (l1 / 2) * np.sin(q[0]) + O[1]
    link[0][0] = l1*np.cos(q[0]) + O[0]
    link[1][0] = l1 * np.sin(q[0]) + O[1]

    com[0][1] = link[0][0] + (l2/2)*np.cos(q[1])
    com[1][1] = link[1][0] + (l2 / 2) * np.sin(q[1])
    link[0][1] = link[0][0] + (l2)*np.cos(q[1])
    link[1][1] = link[1][0] + (l2) * np.sin(q[1])

    com[0][2] = link[0][1] + (l3 / 2) * np.cos(q[2])
    com[1][2] = link[1][1] + (l3 / 2) * np.sin(q[2])
    link[0][2] = link[0][1] + (l3) * np.cos(q[2])
    link[1][2] = link[1][1] + (l3) * np.sin(q[2])

    com[0][3] = link[0][1] - (l4 / 2) * np.cos(q[3])
    com[1][3] = link[1][1] - (l4 / 2) * np.sin(q[3])
    link[0][3] = link[0][1] - (l4) * np.cos(q[3])
    link[1][3] = link[1][1] - (l4) * np.sin(q[3])

    com[0][4] = link[0][3] - (l5 / 2) * np.cos(q[4])
    com[1][4] = link[1][3] - (l5 / 2) * np.sin(q[4])
    link[0][4] = link[0][3] - (l5) * np.cos(q[4])
    link[1][4] = link[1][3] - (l5) * np.sin(q[4])

    return com, link




def stormer_verlet(q0, p0, N, h, O):
    q = np.zeros([5, N])
    qh = np.zeros([5, N])
    p = np.zeros([5, N])
    comx = np.zeros([5, N])
    comy = np.zeros([5, N])
    linkx = np.zeros([5, N])
    linky = np.zeros([5, N])
    comhx = np.zeros([5, N])
    comhy = np.zeros([5, N])
    linkhx = np.zeros([5, N])
    linkhy = np.zeros([5, N])
    #H = np.zeros([N])

    q[0][0] = q0[0]
    q[1][0] = q0[1]
    q[2][0] = q0[2]
    q[3][0] = q0[3]
    q[4][0] = q0[4]

    p[0][0] = p0[0]
    p[1][0] = p0[1]
    p[2][0] = p0[2]
    p[3][0] = p0[3]
    p[4][0] = p0[4]

    # com, link = position(q[:,0], O)
    # comx[:,0] = com[0,:]
    # comy[:, 0] = com[1, :]
    # linkx[:, 0] = link[0, :]
    # linky[:, 0] = link[1, :]
    #
    # KE = (1/2)*(((p[0][0])**2)/m1 + ((p[1][0])**2)/m2 + ((p[2][0])**2)/m3 + ((p[3][0])**2)/m4 + ((p[4][0])**2)/m5)
    # PE = g*(m1*comy[0][0] + m2*comy[1][0] + m3*comy[2][0] + m4*comy[3][0] + m5*comy[4][0])
    # H[0] = KE + PE

    for i in range(N-1):
        com, link = position(q[:, i], O)
        comx[:, i] = com[0, :]
        comy[:, i] = com[1, :]
        linkx[:, i] = link[0,:]
        linky[:, i] = link[1, :]

        rsq = (comx[:, i])**2 + (comy[:, i])**2


        for j in range(5):
            qh[j][i+1] = q[j][i] + (h/2)*p[j][i]/(m[j]*rsq[j])

        p[0][i+1] = p[0][i] + h*(-(m1/2+m2+m3+m4+m5))*l1*g*np.cos(qh[0][i])
        p[1][i + 1] = p[1][i] + h * (-(m2 / 2 + m3 + m4 + m5)) * l2 * g * np.cos(qh[1][i])
        p[2][i + 1] = p[2][i] + h * (-(m3/2)) * l3 * g * np.cos(qh[2][i])
        p[3][i + 1] = p[3][i] + h * (-(m4/2 + m5)) * l4 * g * np.cos(qh[3][i])
        p[4][i + 1] = p[4][i] + h * (-(m5/2)) * l5 * g * np.cos(qh[4][i])

        comh, linkh = position(qh[:, i+1], O)
        comhx[:, i] = comh[0, :]
        comhy[:, i] = comh[1, :]
        linkhx[:, i] = linkh[0, :]
        linkhy[:, i] = linkh[1, :]

        rhsq = (comx[:, i]) ** 2 + (comy[:, i]) ** 2

        for j in range(5):
            q[j][i + 1] = qh[j][i+1] + (h / 2) * p[j][i+1]*(m[j]*rhsq[j])

    H = np.zeros([N])
    KE = np.zeros([N])
    PE = np.zeros([N])

    for i in range(N):
        com, link = position(q[:, i], O)
        comx[:, i] = com[0, :]
        comy[:, i] = com[1, :]
        linkx[:, i] = link[0, :]
        linky[:, i] = link[1, :]

        rsq = (comx[:, i]) ** 2 + (comy[:, i]) ** 2

        KE[i] = (1 / 2) * (
                    ((p[0][i]) ** 2) / (m1*rsq[0]) + ((p[1][i]) ** 2) / (m2*rsq[1]) + ((p[2][i]) ** 2) / (m3*rsq[2]) + ((p[3][i]) ** 2) / (m4*rsq[3]) + (
                        (p[4][i]) ** 2) / (m5*rsq[4]))
        PE[i] = g * (m1 * comy[0][i] + m2 * comy[1][i] + m3 * comy[2][i] + m4 * comy[3][i] + m5 * comy[4][i])
        H[i] = KE[i] + PE[i]

    return q, p, linkx, linky, KE, PE, H

def animation(linkx, linky):
    fig = plt.figure()
    #plt.title(name)
    camera = Camera(fig)
    for i in range(N):
        plt.plot([0, linkx[0, i]], [0, linky[0, i]], 'r')
        plt.plot([linkx[0, i], linkx[1, i]], [linky[0, i], linky[1, i]], 'r')
        plt.plot([linkx[1, i], linkx[2, i]], [linky[1, i], linky[2, i]], 'r')
        plt.plot([linkx[1, i], linkx[3, i]], [linky[1, i], linky[3, i]], 'r')
        plt.plot([linkx[3, i], linkx[4, i]], [linky[3, i], linky[4, i]], 'r')

        camera.snap()
    animation = camera.animate(repeat=False)

    plt.show()
    plt.close()

q, p, linkx, linky, KE, PE, H = stormer_verlet(q0, p0, N, h, O)
t=np.arange(len(H))
plt.plot(h*t, H, label="H")
plt.plot(h*t, KE, label="KE")
plt.plot(h*t, PE, label="PE")
plt.legend()
plt.show()
#animation(linkx, linky)