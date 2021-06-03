import numpy as np

def phiKepler(q0, p0, t, mu):
    h=0.1
    r0 = ((q0[0][0])**2 + (q0[1][0])**2)**(1/2)
    u = q0[0][0]*p0[0][0] + q0[1][0]*p0[1][0]
    E = (1/2)*(p0[0][0]*p0[0][0] + p0[1][0]*p0[1][0]) - mu/r0
    a = -mu/(2*E)
    w = (mu/(a**3))**(1/2)
    sigma = 1 - r0/a
    si = u/(w*(a**2))
    q = np.zeros([2, len(t)])
    p = np.zeros([2, len(t)])
    x = np.zeros(len(t))
    x[0] = w*t[0]*a/r0
    #print(q[1][:])
    q[0][0] = q0[0]
    q[0][1] = q0[1]
    p[0][0] = p0[0]
    p[0][1] = p0[1]
    for i in range(len(t)-1):
        num = x[i] - sigma*np.sin(x[i]) + si*(1-np.cos(x[i])) - w*t[i]
        den = 1 - sigma*np.cos(x[i]) + si*np.sin(x[i])
        x[i+1] = x[i] - (num/den)

    for i in range(len(t)):
        fq = 1 + (np.cos(x[i])-1)*a/r0
        gq = t[i] + (np.sin(x[i])-x[i])/w
        fp = -a*w*np.sin(x[i])/(r0*(1-sigma*np.cos(x[i])+si*np.sin(x[i])))
        gp = 1 + (np.cos(x[i])-1)/(1-sigma*np.cos(x[i])+si*np.sin(x[i]))

        q[0][i] = fq*q0[0][0] + gq*p0[0][0]
        q[1][i] = fq * q0[1][0] + gq * p0[1][0]
        p[0][i] = fp*q0[0][0] + gp*p0[0][0]
        p[1][i] = fp * q0[1][0] + gp * p0[1][0]

    H = np.zeros([len(t)])
    for i in range(len(t)):
        H[i] = ((p[0][i]) ** 2 + (p[1][i]) ** 2) / 2 - mu / (((q[0][i]) ** 2 + (q[1][i]) ** 2) ** (1 / 2))

    return q, p, H
