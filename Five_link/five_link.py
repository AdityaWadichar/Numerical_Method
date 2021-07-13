import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera
from scipy.optimize import fsolve


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
N = 200
h = 0.01
O = [0, 0]


#initial conditions
q10 = np.pi/6
q20 = -np.pi/6
q30 = np.pi/6
q40 = np.pi/6
q50 = np.pi/3

dq10 = -0.1*0
dq20 = 0.1*0
dq30 = -0.1*0
dq40 = 0.1*0
dq50 = -0.1*0


# required declarations
q0 = [q10, q20, q30, q40, q50]
dq0 = [dq10, dq20, dq30, dq40, dq50]


q = np.zeros([5, N])
qh = np.zeros([5, N])
dq = np.zeros([5, N])
dqh = np.zeros([5, N])
comx = np.zeros([5, N])
comy = np.zeros([5, N])
linkx = np.zeros([5, N])
linky = np.zeros([5, N])

q[0][0] = q0[0]
q[1][0] = q0[1]
q[2][0] = q0[2]
q[3][0] = q0[3]
q[4][0] = q0[4]

dq[0][0] = dq0[0]
dq[1][0] = dq0[1]
dq[2][0] = dq0[2]
dq[3][0] = dq0[3]
dq[4][0] = dq0[4]

TE = np.zeros([N])
KE = np.zeros([N])
PE = np.zeros([N])


def position(q, O):
    com = np.zeros([2, 5])
    link = np.zeros([2, 5])

    com[0][0] = (l1/2) * np.sin(q[0]) + O[0]
    com[1][0] = (l1 / 2) * np.cos(q[0]) + O[1]
    link[0][0] = l1*np.sin(q[0]) + O[0]
    link[1][0] = l1 * np.cos(q[0]) + O[1]

    com[0][1] = link[0][0] + (l2/2)*np.sin(q[1])
    com[1][1] = link[1][0] + (l2 / 2) * np.cos(q[1])
    link[0][1] = link[0][0] + (l2)*np.sin(q[1])
    link[1][1] = link[1][0] + (l2) * np.cos(q[1])

    com[0][2] = link[0][1] + (l3 / 2) * np.sin(q[2])
    com[1][2] = link[1][1] + (l3 / 2) * np.cos(q[2])
    link[0][2] = link[0][1] + (l3) * np.sin(q[2])
    link[1][2] = link[1][1] + (l3) * np.cos(q[2])

    com[0][3] = link[0][1] - (l4 / 2) * np.sin(q[3])
    com[1][3] = link[1][1] - (l4 / 2) * np.cos(q[3])
    link[0][3] = link[0][1] - (l4) * np.sin(q[3])
    link[1][3] = link[1][1] - (l4) * np.cos(q[3])

    com[0][4] = link[0][3] - (l5 / 2) * np.sin(q[4])
    com[1][4] = link[1][3] - (l5 / 2) * np.cos(q[4])
    link[0][4] = link[0][3] - (l5) * np.sin(q[4])
    link[1][4] = link[1][3] - (l5) * np.cos(q[4])

    return com, link



def dynamics(q, dq):
    q1, q2, q3, q4, q5 = q[:]
    dq1, dq2, dq3, dq4, dq5 = dq[:]


    # first link

    a11 = ((m1/3) + m2 + m3 + m4 + m5)*(l1**2)
    a12 = ((m2/2) + m3 + m4 + m5)*l1*l2*np.cos(q1-q2)
    a13 = (1/2)*m3*l1*l3*np.cos(q1-q3)
    a14 = -((m4/2) + m5)*l1*l4*np.cos(q1-q4)
    a15 = -(1/2)*m5*l1*l5*np.cos(q1-q5)

    b11 = 0
    b12 = -((m2/2) + m3 + m4 + m5)*l1*l2*(np.sin(q1-q2))*(dq1-dq2)*dq2
    #print(dq1, dq2)
    b13 = -(m3/2)*l1*l3*(np.sin(q1-q3))*(dq1-dq3)*dq3
    b14 = ((m4/2) + m5)*l1*l4*(np.sin(q1-q4))*(dq1-dq4)*dq4
    b15 = (m5/2)*l1*l5*(np.sin(q1-q5))*(dq1-dq5)*dq5
    b1 = b11 + b12 + b13 + b14 + b15

    c11 = ((m1/2) + m2 + m3 + m4 + m5)*g*l1*np.sin(q1)
    c12 = -((m2/2) + m3 + m4 + m5)*l1*l2*dq1*dq2*np.sin(q1-q2)
    c13 = -(m3/2)*l1*l3*dq1*dq3*np.sin(q1-q3)
    c14 = ((m4/2) + m5)*l1*l4*dq1*dq4*np.sin(q1-q4)
    c15 = (m5/2)*l1*l5*dq1*dq5*np.sin(q1-q5)
    c1 = c11 + c12 + c13 + c14 + c15

    # second link

    a21 = ((m2/2) + m3 + m4 + m5)*l1*l2*np.cos(q1-q2)
    a22 = ((m2/3) + m3 + m4 + m5)*(l2**2)
    a23 = (m3/2)*l2*l3*np.cos(q2-q3)
    a24 = -((m4/2) + m5)*l2*l4*np.cos(q2-q4)
    a25 = -(m5/2)*l2*l5*np.cos(q2-q5)

    b21 = -((m2/2) + m3 + m4 + m5)*l1*l2*(np.sin(q1-q2))*(dq1-dq2)*dq1
    b22 = 0
    b23 = -(m3/2)*l2*l3*(np.sin(q2-q3))*(dq2-dq3)*dq3
    b24 = ((m4/2) + m5)*l2*l4*(np.sin(q2-q4))*(dq2-dq4)*dq4
    b25 = (m5/2)*l2*l5*(np.sin(q2-q5))*(dq2-dq5)*dq5
    b2 = b21 + b22 + b23 + b24 + b25

    c21 = ((m2/2) + m3 + m4 + m5)*l1*l2*dq1*dq2*np.sin(q1-q2)
    c22 = ((m2/2) + m3 + m4 + m5)*g*l2*np.sin(q2)
    c23 = -(m3/2)*l2*l3*dq2*dq3*np.sin(q2-q3)
    c24 = ((m4/2) + m5)*l2*l4*dq2*dq4*np.sin(q2-q4)
    c25 = (m5/2)*l2*l5*dq2*dq5*np.sin(q2-q5)
    c2 = c21 + c22 + c23 + c24 + c25


    # third link

    a31 = (m3/2)*l1*l3*np.cos(q1-q3)
    a32 = (m3/2)*l2*l3*np.cos(q2-q3)
    a33 = (m3/3)*(l3**2)
    a34 = 0
    a35 = 0

    b31 = (m3/2)*l1*l3*(np.sin(q1-q3))*(dq1-dq3)*dq1
    b32 = (m3/2)*l2*l3*(np.sin(q2-q3))*(dq2-dq3)*dq2
    b33=0
    b34=0
    b35=0
    b3 = b31 + b32 + b33 + b34 + b35

    c33 = (m3/2)*g*l3*np.sin(q3)
    c32 = (m3/2)*l2*l3*dq2*dq3*np.sin(q2-q3)
    c31 = (m3/2)*l1*l3*dq1*dq3*np.sin(q1-q3)
    c34=0
    c35=0
    c3 = c31 + c32 + c33 + c34 + c35

    # fourth link

    a41 = -((m4/2) + m5)*l1*l4*np.cos(q1-q4)
    a42 = -((m4/2) + m5)*l2*l4*np.cos(q2-q4)
    a43 = 0
    a44 = ((m4/3) + m5)*(l4**2)
    a45 = (m5/2)*l4*l5*np.cos(q4-q5)

    b41 = ((m4/2) + m5)*l1*l4*(np.sin(q1-q4))*(dq1-dq4)*dq1
    b42 = ((m4/2) + m5)*l2*l4*(np.sin(q2-q4))*(dq2-dq4)*dq2
    b43 = 0
    b44 = 0
    b45 = -(m5/2)*l4*l5*(np.sin(q4-q5))*(dq4-dq5)*dq5
    b4 = b41 + b42 + b43 + b44 + b45

    c44 = -((m4/2) + m5)*g*l4*np.sin(q4)
    c41 = -((m4/2) + m5)*l1*l4*dq1*dq4*np.sin(q1-q4)
    c42 = -((m4/2) + m5)*l2*l4*dq1*dq4*np.sin(q2-q4)
    c43 = 0
    c45 = -(m5/2)*l4*l5*dq4*dq5*np.sin(q4-q5)
    c4 = c41 + c42 + c43 + c44 + c45


    # fifth link

    a51 = -(m5/2)*l1*l5*np.cos(q1-q5)
    a52 = -(m5/2)*l2*l5*np.cos(q2-q5)
    a53 = 0
    a54 = (m5/2)*l4*l5*np.cos(q4-q5)
    a55 = (m5/3)*(l5**2)

    b51 = (m5/2)*l1*l5*(np.sin(q1-q5))*(dq1-dq5)*dq1
    b52 = (m5/2)*l2*l5*(np.sin(q2-q5))*(dq2-dq5)*dq2
    b53 = 0
    b54 = -(m5/2)*l4*l5*(np.sin(q4-q5))*(dq4-dq5)*dq4
    b55 = 0
    b5 = b51 + b52 + b53 + b54 + b55

    c51 = -(m5/2)*l1*l5*dq1*dq5*np.sin(q1-q5)
    c52 = -(m5/2)*l2*l5*dq2*dq5*np.sin(q2-q5)
    c53 = 0
    c54 = (m5/2)*l4*l5*dq4*dq5*np.sin(q4-q5)
    c55 = 0
    c5 = c51 + c52 + c53 + c54 + c55


    # matrix equations

    A = np.matrix(((a11, a12, a13, a14, a15), (a21, a22, a23, a24, a25), (a31, a32, a33, a34, a35), (a41, a42, a43, a44, a45), (a51, a52, a53, a54, a55)))
    B = np.matrix((b1, b2, b3, b4, b5))
    B = B.transpose()
    C = np.matrix((c1, c2, c3, c4, c5))
    C = C.transpose()
    T = np.matrix((1, 1, 1, 1, 1))*0
    T = T.transpose()

    X = (A**(-1))*(T+C-B)
    X=X.transpose()

    return X



def dynamics1(q, dq):
    q1, q2, q3, q4, q5 = q[:]
    dq1, dq2, dq3, dq4, dq5 = dq[:]

    n=[m5*l1, m5*l2, 0, m5*l5, 0]

    A=np.matrix(((m1*(l1**2)/3 + n[0]*l1, m2*l2*l1/2 + m2*l1, m3*l3*l1/2 + m3*l1, m4*l4*l1/2 + m4*l1, m5*l5*l1/2 + m5*l1),
                 (m1*l1*l2/2 + m1*l2, m2*(l2**2)/3 + n[1]*l2, m3*l3*l2/2 + m3*l2, m4*l4*l2/2 + m4*l2, m5*l5*l2/2 + m5*l2),
                 (m1*l1*l3/2 + m1*l3, m2*l2*l3/2 + m2*l3, m3*(l3**2)/3 + n[2]*l3, 0, 0),
                 (m1*l1*l4/2 + m1*l4, m2*l2*l4/2 + m2*l4, m3*l3*l4/2 + m3*l4, m4*(l4**2)/3 + n[3]*l4, m5*l5*l4/2 + m5*l4),
                 (m1*l1*l5/2 + m1*l5, m2*l2*l5/2 + m2*l5, m3*l3*l5/2 + m3*l5, m4*l4*l5/2 + m4*l5, m5*(l5**2)/3 + n[4]*l5)))

    B=np.matrix(((q1-q1, q1-q2, q1-q3, q1+q4, q1+q5),
                 (q2-q1, q2-q2, q2-q3, q2+q4, q2+q5),
                 (q3-q1, q3-q2, q3-q3, q3+q4, q3+q5),
                 (q4+q1, q4+q2, q4+q3, q4-q4, q4-q5),
                 (q5+q1, q5+q2, q5+q3, q5-q4, q5-q5)))

    D=np.multiply(A, np.cos(B))
    H=np.multiply(A, np.sin(B))

    G=np.matrix((((m1*l1/2 + n[0])*np.sin(q1)),
                 ((m2*l2/2 + n[1])*np.sin(q2)),
                 ((m3*l3/2 + n[2])*np.sin(q3)),
                 (-(m4*l4/2 + n[3])*np.sin(q4)),
                 (-(m5*l5/2 + n[4])*np.sin(q5))))
    G=G*(-g)
    G=G.transpose()

    T = np.matrix((0, 0, 0, 0, 0))
    T=T.transpose()
    sdq = np.matrix((dq1**2, dq2**2, dq3**2, dq4**2, dq5**2))
    sdq=sdq.transpose()

    ddq = (D**(-1))*(T-G-H*sdq)
    ddq=ddq.transpose()
    return ddq


def equation(dqh1):
    #dq1, dq2, dq3, dq4, dq5 = dqh1
    if (Dynamics==0):
        ddq = dynamics(q[:, current], dqh1)
    else:
        ddq = dynamics1(q[:, current], dqh1)
    ddq1 = np.matrix((ddq[0,0], ddq[0,1], ddq[0,2], ddq[0,3], ddq[0,4]))
    a=dqh1 - dq[:, current] - (h/2)*ddq1
    b=[a[0,0], a[0,1], a[0,2], a[0,3], a[0,4]]

    return b

def equation3(dq1):
    #dq1, dq2, dq3, dq4, dq5 = dqh1
    if (Dynamics==0):
        ddq = dynamics(q[:, current+1], dq1)
    else:
        ddq = dynamics1(q[:, current+1], dq1)
    ddq1 = np.matrix((ddq[0,0], ddq[0,1], ddq[0,2], ddq[0,3], ddq[0,4]))
    a=dq1 - dqh[:, current+1] - (h/2)*ddq1
    b=[a[0,0], a[0,1], a[0,2], a[0,3], a[0,4]]

    return b

def equation_1(dqn1):
    if (Dynamics==0):
        ddq =  dynamics(qh[:, current+1], dq[:, current])
        ddqn = dynamics(qh[:, current+1], dqn1)
    else:
        ddq = dynamics1(qh[:, current + 1], dq[:, current])
        ddqn = dynamics1(qh[:, current + 1], dqn1)

    a= dqn1 - dq[:, current] - (h/2)*(ddqn + ddqn)
    b = [a[0, 0], a[0, 1], a[0, 2], a[0, 3], a[0, 4]]

    return b


def stormer_verlet(q0, dq0, O):

    # no equations with dq(n+1/2)

    global q, dqh, dq, comx, comy, linkx, linky, TE, KE, PE, current, h, N, Dynamics


    for current in range(N-1):
        i = current
        if (Dynamics==0):
            ddq = dynamics(q[:, i], dq[:, i])
        else:
            ddq = dynamics1(q[:, i], dq[:, i])
        dqh[:, i+1] = dq[:, i] + (h/2)*ddq

        q[:, i+1] = q[:, i] + h*dqh[:, i+1]

        ddq = dynamics(q[:, i+1], dqh[:, i+1])
        dq[:, i+1] = dqh[:, i+1] + (h/2)*ddq



    for i in range(N):
        com, link = position(q[:, i], O)
        comx[:, i] = com[0, :]
        comy[:, i] = com[1, :]
        linkx[:, i] = link[0, :]
        linky[:, i] = link[1, :]
        q1, q2, q3, q4, q5 = q[:, i]
        dq1, dq2, dq3, dq4, dq5 = dq[:, i]

        KE1 = (m1/6)*(l1**2)*(dq1**2)
        KE2 = (m2/2)*((l1**2)*(dq1**2) + (1/3)*(l2**2)*(dq2**2) + l1*l2*dq1*dq2*np.cos(q1-q2))
        KE3 = (m3/2)*((l1**2)*(dq1**2) + (l2**2)*(dq2**2) + (1/4)*(l3**2)*(dq3**2) + 2*l1*l2*dq1*dq2*np.cos(q1-q2)
                      + l2*l3*dq2*dq3*np.cos(q2-q3) + l1*l3*dq1*dq3*np.cos(q1-q3)) + (1/24)*m3*(l3**2)*(dq3**2)

        KE4 = (m4/2)*((l1**2)*(dq1**2) + (l2**2)*(dq2**2) + (1/4)*(l4**2)*(dq4**2) + 2*l1*l2*dq1*dq2*np.cos(q1-q2)
                      - l2*l4*dq2*dq4*np.cos(q2-q4) - l1*l4*dq1*dq4*np.cos(q1-q4)) + (1/24)*m4*(l4**2)*(dq4**2)

        KE5 = (m5/2)*((l1**2)*(dq1**2) + (l2**2)*(dq2**2) + (l4**2)*(dq4**2) + (1/4)*(l5**2)*(dq5**2) + 2*l1*l2*dq1*dq2*np.cos(q1-q2)
                      - 2*l2*l4*dq2*dq4*np.cos(q2-q4) - 2*l1*l4*dq1*dq4*np.cos(q1-q4) - l1*l5*dq1*dq5*np.cos(q1-q5)
                      - l2*l5*dq2*dq5*np.cos(q2-q5) + l4*l5*dq4*dq5*np.cos(q4-q5)) + (1/24)*m5*(l5**2)*(dq5**2)

        KE[i] = KE1 + KE2 + KE3 + KE4 + KE5


        # KE[i] = (1 / 2) * (
        #             ((p[0][i]) ** 2) / (m1*rsq[0]) + ((p[1][i]) ** 2) / (m2*rsq[1]) + ((p[2][i]) ** 2) / (m3*rsq[2]) + ((p[3][i]) ** 2) / (m4*rsq[3]) + (
        #                 (p[4][i]) ** 2) / (m5*rsq[4]))

        PE[i] = g * (m1 * comy[0][i] + m2 * comy[1][i] + m3 * comy[2][i] + m4 * comy[3][i] + m5 * comy[4][i])
        TE[i] = KE[i] + PE[i]

    return



def stormer_verlet1(q0, dq0, O):

    # no equations with q(n+1/2)

    global q, dqh, dq, comx, comy, linkx, linky, TE, KE, PE, current, h, N, Dynamics


    for current in range(N-1):
        i = current

        qh[:, i+1] = q[:, i] + (h/2)*dq[:, i]
        if (Dynamics==0):
            ddq = dynamics(qh[:, i+1], dq[:, i])
        else:
            ddq = dynamics1(qh[:, i + 1], dq[:, i])
        dq[:, i+1] = dq[:, i] + h*ddq
        q[:, i+1] = qh[:, i+1] + (h/2)*dq[:, i+1]




    for i in range(N):
        com, link = position(q[:, i], O)
        comx[:, i] = com[0, :]
        comy[:, i] = com[1, :]
        linkx[:, i] = link[0, :]
        linky[:, i] = link[1, :]
        q1, q2, q3, q4, q5 = q[:, i]
        dq1, dq2, dq3, dq4, dq5 = dq[:, i]

        KE1 = (m1/6)*(l1**2)*(dq1**2)
        KE2 = (m2/2)*((l1**2)*(dq1**2) + (1/3)*(l2**2)*(dq2**2) + l1*l2*dq1*dq2*np.cos(q1-q2))
        KE3 = (m3/2)*((l1**2)*(dq1**2) + (l2**2)*(dq2**2) + (1/4)*(l3**2)*(dq3**2) + 2*l1*l2*dq1*dq2*np.cos(q1-q2)
                      + l2*l3*dq2*dq3*np.cos(q2-q3) + l1*l3*dq1*dq3*np.cos(q1-q3)) + (1/24)*m3*(l3**2)*(dq3**2)

        KE4 = (m4/2)*((l1**2)*(dq1**2) + (l2**2)*(dq2**2) + (1/4)*(l4**2)*(dq4**2) + 2*l1*l2*dq1*dq2*np.cos(q1-q2)
                      - l2*l4*dq2*dq4*np.cos(q2-q4) - l1*l4*dq1*dq4*np.cos(q1-q4)) + (1/24)*m4*(l4**2)*(dq4**2)

        KE5 = (m5/2)*((l1**2)*(dq1**2) + (l2**2)*(dq2**2) + (l4**2)*(dq4**2) + (1/4)*(l5**2)*(dq5**2) + 2*l1*l2*dq1*dq2*np.cos(q1-q2)
                      - 2*l2*l4*dq2*dq4*np.cos(q2-q4) - 2*l1*l4*dq1*dq4*np.cos(q1-q4) - l1*l5*dq1*dq5*np.cos(q1-q5)
                      - l2*l5*dq2*dq5*np.cos(q2-q5) + l4*l5*dq4*dq5*np.cos(q4-q5)) + (1/24)*m5*(l5**2)*(dq5**2)

        KE[i] = KE1 + KE2 + KE3 + KE4 + KE5


        # KE[i] = (1 / 2) * (
        #             ((p[0][i]) ** 2) / (m1*rsq[0]) + ((p[1][i]) ** 2) / (m2*rsq[1]) + ((p[2][i]) ** 2) / (m3*rsq[2]) + ((p[3][i]) ** 2) / (m4*rsq[3]) + (
        #                 (p[4][i]) ** 2) / (m5*rsq[4]))

        PE[i] = g * (m1 * comy[0][i] + m2 * comy[1][i] + m3 * comy[2][i] + m4 * comy[3][i] + m5 * comy[4][i])
        TE[i] = KE[i] + PE[i]

    return





def stormer_verlet2(q0, dq0, O):

    # equations with dq(n+1/2)

    global q, dqh, dq, comx, comy, linkx, linky, TE, KE, PE, current, h, N, Dynamics


    for current in range(N-1):
        i = current

        dq1, dq2, dq3, dq4, dq5 = fsolve(equation, (1, 1, 1, 1, 1))
        dqh[:, i+1] = [dq1, dq2, dq3, dq4, dq5]

        q[:, i+1] = q[:, i] + h*dqh[:, i+1]

        # ddq = dynamics1(q[:, i+1], dqh[:, i+1])
        # dq[:, i+1] = dqh[:, i+1] + (h/2)*ddq

        dq1, dq2, dq3, dq4, dq5 = fsolve(equation3, (1, 1, 1, 1, 1))
        dq[:, i + 1] = [dq1, dq2, dq3, dq4, dq5]





    for i in range(N):
        com, link = position(q[:, i], O)
        comx[:, i] = com[0, :]
        comy[:, i] = com[1, :]
        linkx[:, i] = link[0, :]
        linky[:, i] = link[1, :]
        q1, q2, q3, q4, q5 = q[:, i]
        dq1, dq2, dq3, dq4, dq5 = dq[:, i]

        KE1 = (m1/6)*(l1**2)*(dq1**2)
        KE2 = (m2/2)*((l1**2)*(dq1**2) + (1/3)*(l2**2)*(dq2**2) + l1*l2*dq1*dq2*np.cos(q1-q2))
        KE3 = (m3/2)*((l1**2)*(dq1**2) + (l2**2)*(dq2**2) + (1/4)*(l3**2)*(dq3**2) + 2*l1*l2*dq1*dq2*np.cos(q1-q2)
                      + l2*l3*dq2*dq3*np.cos(q2-q3) + l1*l3*dq1*dq3*np.cos(q1-q3)) + (1/24)*m3*(l3**2)*(dq3**2)

        KE4 = (m4/2)*((l1**2)*(dq1**2) + (l2**2)*(dq2**2) + (1/4)*(l4**2)*(dq4**2) + 2*l1*l2*dq1*dq2*np.cos(q1-q2)
                      - l2*l4*dq2*dq4*np.cos(q2-q4) - l1*l4*dq1*dq4*np.cos(q1-q4)) + (1/24)*m4*(l4**2)*(dq4**2)

        KE5 = (m5/2)*((l1**2)*(dq1**2) + (l2**2)*(dq2**2) + (l4**2)*(dq4**2) + (1/4)*(l5**2)*(dq5**2) + 2*l1*l2*dq1*dq2*np.cos(q1-q2)
                      - 2*l2*l4*dq2*dq4*np.cos(q2-q4) - 2*l1*l4*dq1*dq4*np.cos(q1-q4) - l1*l5*dq1*dq5*np.cos(q1-q5)
                      - l2*l5*dq2*dq5*np.cos(q2-q5) + l4*l5*dq4*dq5*np.cos(q4-q5)) + (1/24)*m5*(l5**2)*(dq5**2)

        KE[i] = KE1 + KE2 + KE3 + KE4 + KE5


        # KE[i] = (1 / 2) * (
        #             ((p[0][i]) ** 2) / (m1*rsq[0]) + ((p[1][i]) ** 2) / (m2*rsq[1]) + ((p[2][i]) ** 2) / (m3*rsq[2]) + ((p[3][i]) ** 2) / (m4*rsq[3]) + (
        #                 (p[4][i]) ** 2) / (m5*rsq[4]))

        PE[i] = g * (m1 * comy[0][i] + m2 * comy[1][i] + m3 * comy[2][i] + m4 * comy[3][i] + m5 * comy[4][i])
        TE[i] = KE[i] + PE[i]

    return


def stormer_verlet3(q0, dq0, O):

    # equations with q(n+1/2)

    global q, qh, dq, dqh, comx, comy, linkx, linky, TE, KE, PE, current, Dynamics


    for current in range(N-1):
        i = current

        qh[:, i+1] = q[:, i] + (h/2)*dq[:, i]

        dq1, dq2, dq3, dq4, dq5 = fsolve(equation_1, (1, 1, 1, 1, 1))
        dq[:, i + 1] = [dq1, dq2, dq3, dq4, dq5]

        q[:, i+1] = qh[:, i+1] + (h/2)*dq[:, i+1]

    for i in range(N):
        com, link = position(q[:, i], O)
        comx[:, i] = com[0, :]
        comy[:, i] = com[1, :]
        linkx[:, i] = link[0, :]
        linky[:, i] = link[1, :]
        q1, q2, q3, q4, q5 = q[:, i]
        dq1, dq2, dq3, dq4, dq5 = dq[:, i]

        KE1 = (m1/6)*(l1**2)*(dq1**2)
        KE2 = (m2/2)*((l1**2)*(dq1**2) + (1/3)*(l2**2)*(dq2**2) + l1*l2*dq1*dq2*np.cos(q1-q2))
        KE3 = (m3/2)*((l1**2)*(dq1**2) + (l2**2)*(dq2**2) + (1/4)*(l3**2)*(dq3**2) + 2*l1*l2*dq1*dq2*np.cos(q1-q2)
                      + l2*l3*dq2*dq3*np.cos(q2-q3) + l1*l3*dq1*dq3*np.cos(q1-q3)) + (1/24)*m3*(l3**2)*(dq3**2)

        KE4 = (m4/2)*((l1**2)*(dq1**2) + (l2**2)*(dq2**2) + (1/4)*(l4**2)*(dq4**2) + 2*l1*l2*dq1*dq2*np.cos(q1-q2)
                      - l2*l4*dq2*dq4*np.cos(q2-q4) - l1*l4*dq1*dq4*np.cos(q1-q4)) + (1/24)*m4*(l4**2)*(dq4**2)

        KE5 = (m5/2)*((l1**2)*(dq1**2) + (l2**2)*(dq2**2) + (l4**2)*(dq4**2) + (1/4)*(l5**2)*(dq5**2) + 2*l1*l2*dq1*dq2*np.cos(q1-q2)
                      - 2*l2*l4*dq2*dq4*np.cos(q2-q4) - 2*l1*l4*dq1*dq4*np.cos(q1-q4) - l1*l5*dq1*dq5*np.cos(q1-q5)
                      - l2*l5*dq2*dq5*np.cos(q2-q5) + l4*l5*dq4*dq5*np.cos(q4-q5)) + (1/24)*m5*(l5**2)*(dq5**2)

        KE[i] = KE1 + KE2 + KE3 + KE4 + KE5


        # KE[i] = (1 / 2) * (
        #             ((p[0][i]) ** 2) / (m1*rsq[0]) + ((p[1][i]) ** 2) / (m2*rsq[1]) + ((p[2][i]) ** 2) / (m3*rsq[2]) + ((p[3][i]) ** 2) / (m4*rsq[3]) + (
        #                 (p[4][i]) ** 2) / (m5*rsq[4]))

        PE[i] = g * (m1 * comy[0][i] + m2 * comy[1][i] + m3 * comy[2][i] + m4 * comy[3][i] + m5 * comy[4][i])
        TE[i] = KE[i] + PE[i]


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
    animation = camera.animate(repeat=False, interval=60)

    plt.show()
    plt.close()


Dynamics = 0
stormer_verlet2(q0, dq0, O)
#print(dq)
t=np.arange(N)
plt.plot(h*t, TE, label="Total Energy")
plt.plot(h*t, KE, label="Kinetic Energy")
plt.plot(h*t, PE, label="Potential Energy")
plt.xlabel("t")
plt.ylabel("Energy")
plt.legend()
plt.show()
animation(linkx, linky)