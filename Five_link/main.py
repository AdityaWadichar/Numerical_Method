import casadi as ca
import numpy as np
from matplotlib import pyplot as plt


# import cv2
# from PIL import Image

class walker():
    def __init__(self):
        # set our parameters of optimization
        self.opti = ca.Opti()
        self.N = 100
        self.T = 0.1
        self.step_max = 1
        self.tauMax = 1.5
        self.pi = np.pi
        self.l1 = 0.5
        self.l2 = 0.5
        self.l3 = 0.6
        self.m = [1, 1, 1.5, 1, 1]
        #self.m = [0.5, 0.5, 0.5, 0.5, 0.5]  # ; self.m2 = 0.5; self.m3 = 0.5
        self.i1 = self.m[0] * (self.l1 ** 2) / 12
        self.i2 = self.m[1] * (self.l2 ** 2) / 12
        self.i3 = self.m[2] * (self.l3 ** 2) / 12
        self.i = [self.i1, self.i2, self.i3, self.i2, self.i1]
        self.g = -9.81
        self.h = self.T / self.N
        goali = [-0.3, 0.5, 0.0, -0.5, -0.3]
        goalf = goali[::-1]

        self.goal = [goali, goalf]
        # set our optimization variables
        self.state = []
        self.halfstate=[]
        self.u = []
        for i in range(self.N):
            rowu = []
            rowq = []
            rowdq12 = []
            rowq.append(self.opti.variable(5))
            rowq.append(self.opti.variable(5))
            rowdq12.append(self.opti.variable(5))
            rowu.append(self.opti.variable(4))
            self.state.append(rowq)
            self.halfstate.append(rowdq12)
            self.u.append(rowu)

        self.pos = []
        self.com = []
        self.ddq = []
        self.ddq12 = []
        for i in range(self.N):
            p, dp, g, dg, ddq = self.getModel(self.state[i], self.u[i])
            self.pos.append(p)
            self.com.append(g)
            self.ddq.append(ddq)
            if (i<self.N-1):
                p12, dp12, g12, dg12, ddq12 = self.getModel12(self.state[i], self.halfstate[i], self.u[i])
                self.ddq12.append(ddq12)
            if i == 0:
                self.dp0 = dp
            if i == self.N - 1:
                self.dpN = dp
                self.impactmap = self.heelStrike(self.state[i][0], self.state[i][1], p, dp, g, dg)

    def getModel(self, state, u):
        q = state[0]
        dq = state[1]
        u = u[0]
        p, dp, g, dc = self.getKinematics(q, dq)

        ddc10, ddc11 = ca.jtimes(dc[0][0], dq, dq), ca.jtimes(dc[0][1], dq, dq)
        ddc20, ddc21 = ca.jtimes(dc[0][0], dq, dq), ca.jtimes(dc[0][1], dq, dq)
        ddc30, ddc31 = ca.jtimes(dc[0][0], dq, dq), ca.jtimes(dc[0][1], dq, dq)
        ddc40, ddc41 = ca.jtimes(dc[0][0], dq, dq), ca.jtimes(dc[0][1], dq, dq)
        ddc50, ddc51 = ca.jtimes(dc[0][0], dq, dq), ca.jtimes(dc[0][1], dq, dq)
        ddc = [[ddc10, ddc11], [ddc20, ddc21], [ddc30, ddc31], [ddc40, ddc41], [ddc50, ddc51]]
        s = [0, 0, 0, 0, 0]
        for i in range(5):
            s[0] += (((g[i][0]) * self.m[i] * self.g)) + (
                        ((g[i][0]) * self.m[i] * ddc[i][1]) - ((g[i][1]) * self.m[i] * ddc[i][0]))
            if i > 0:
                s[1] += (((g[i][0] - p[0][0]) * self.m[i] * self.g)) + (
                            ((g[i][0] - p[0][0]) * self.m[i] * ddc[i][1]) - (
                                (g[i][1] - p[0][1]) * self.m[i] * ddc[i][0]))
                if i > 1:
                    s[2] += (((g[i][0] - p[1][0]) * self.m[i] * self.g)) + (
                                ((g[i][0] - p[1][0]) * self.m[i] * ddc[i][1]) - (
                                    (g[i][1] - p[1][1]) * self.m[i] * ddc[i][0]))
                    if i > 2:
                        s[3] += (((g[i][0] - p[1][0]) * self.m[i] * self.g)) + (
                                    ((g[i][0] - p[1][0]) * self.m[i] * ddc[i][1]) - (
                                        (g[i][1] - p[1][1]) * self.m[i] * ddc[i][0]))
                        if i > 3:
                            s[4] += (((g[i][0] - p[3][0]) * self.m[i] * self.g)) + (
                                        ((g[i][0] - p[3][0]) * self.m[i] * ddc[i][1]) - (
                                            (g[i][1] - p[3][1]) * self.m[i] * ddc[i][0]))

        ddq5 = (u[3] - s[4]) / self.i1
        ddq4 = ((u[2] - s[3]) - (self.i1 * ddq5)) / self.i2
        ddq3 = ((u[1] - s[2]) - (self.i1 * ddq5) - (self.i2 * ddq4)) / self.i3
        ddq2 = ((u[0] - s[1]) - (self.i1 * ddq5) - (self.i2 * ddq4) - (self.i3 * ddq3)) / self.i2
        ddq1 = ((-s[0]) - (self.i1 * ddq5) - (self.i2 * ddq4) - (self.i3 * ddq3) - (self.i2 * ddq4)) / self.i1
        ddq = [ddq1, ddq2, ddq3, ddq4, ddq5]

        return p, dp, g, dc, ddq

    def getModel12(self, state, dq12, u):
        q = state[0]
        dq = dq12[0]
        u = u[0]
        p, dp, g, dc = self.getKinematics(q, dq)

        ddc10, ddc11 = ca.jtimes(dc[0][0], dq, dq), ca.jtimes(dc[0][1], dq, dq)
        ddc20, ddc21 = ca.jtimes(dc[0][0], dq, dq), ca.jtimes(dc[0][1], dq, dq)
        ddc30, ddc31 = ca.jtimes(dc[0][0], dq, dq), ca.jtimes(dc[0][1], dq, dq)
        ddc40, ddc41 = ca.jtimes(dc[0][0], dq, dq), ca.jtimes(dc[0][1], dq, dq)
        ddc50, ddc51 = ca.jtimes(dc[0][0], dq, dq), ca.jtimes(dc[0][1], dq, dq)
        ddc = [[ddc10, ddc11], [ddc20, ddc21], [ddc30, ddc31], [ddc40, ddc41], [ddc50, ddc51]]
        s = [0, 0, 0, 0, 0]
        for i in range(5):
            s[0] += (((g[i][0]) * self.m[i] * self.g)) + (
                        ((g[i][0]) * self.m[i] * ddc[i][1]) - ((g[i][1]) * self.m[i] * ddc[i][0]))
            if i > 0:
                s[1] += (((g[i][0] - p[0][0]) * self.m[i] * self.g)) + (
                            ((g[i][0] - p[0][0]) * self.m[i] * ddc[i][1]) - (
                                (g[i][1] - p[0][1]) * self.m[i] * ddc[i][0]))
                if i > 1:
                    s[2] += (((g[i][0] - p[1][0]) * self.m[i] * self.g)) + (
                                ((g[i][0] - p[1][0]) * self.m[i] * ddc[i][1]) - (
                                    (g[i][1] - p[1][1]) * self.m[i] * ddc[i][0]))
                    if i > 2:
                        s[3] += (((g[i][0] - p[1][0]) * self.m[i] * self.g)) + (
                                    ((g[i][0] - p[1][0]) * self.m[i] * ddc[i][1]) - (
                                        (g[i][1] - p[1][1]) * self.m[i] * ddc[i][0]))
                        if i > 3:
                            s[4] += (((g[i][0] - p[3][0]) * self.m[i] * self.g)) + (
                                        ((g[i][0] - p[3][0]) * self.m[i] * ddc[i][1]) - (
                                            (g[i][1] - p[3][1]) * self.m[i] * ddc[i][0]))

        ddq5 = (u[3] - s[4]) / self.i1
        ddq4 = ((u[2] - s[3]) - (self.i1 * ddq5)) / self.i2
        ddq3 = ((u[1] - s[2]) - (self.i1 * ddq5) - (self.i2 * ddq4)) / self.i3
        ddq2 = ((u[0] - s[1]) - (self.i1 * ddq5) - (self.i2 * ddq4) - (self.i3 * ddq3)) / self.i2
        ddq1 = ((-s[0]) - (self.i1 * ddq5) - (self.i2 * ddq4) - (self.i3 * ddq3) - (self.i2 * ddq4)) / self.i1
        ddq = [ddq1, ddq2, ddq3, ddq4, ddq5]

        return p, dp, g, dc, ddq

    def getKinematics(self, q, dq):
        p10 = ca.MX.sym('p10', 1);
        c10 = ca.MX.sym('g10', 1)
        p11 = ca.MX.sym('p11', 1);
        c11 = ca.MX.sym('g11', 1)
        p20 = ca.MX.sym('p20', 1);
        c20 = ca.MX.sym('g20', 1)
        p21 = ca.MX.sym('p21', 1);
        c21 = ca.MX.sym('g21', 1)
        p30 = ca.MX.sym('p30', 1);
        c30 = ca.MX.sym('g30', 1)
        p31 = ca.MX.sym('p31', 1);
        c31 = ca.MX.sym('g31', 1)
        p40 = ca.MX.sym('p40', 1);
        c40 = ca.MX.sym('g40', 1)
        p41 = ca.MX.sym('p41', 1);
        c41 = ca.MX.sym('g41', 1)
        p50 = ca.MX.sym('p50', 1);
        c50 = ca.MX.sym('g50', 1)
        p51 = ca.MX.sym('p51', 1);
        c51 = ca.MX.sym('g51', 1)

        p0 = [0, 0]

        p10, p11 = -self.l1 * ca.sin(q[0]) + p0[0], self.l1 * ca.cos(q[0]) + p0[0]
        p20, p21 = -self.l2 * ca.sin(q[1]) + p10, self.l2 * ca.cos(q[1]) + p11
        p30, p31 = self.l3 * ca.sin(q[2]) + p20, self.l3 * ca.cos(q[2]) + p21
        p40, p41 = (self.l2 * ca.sin(q[3])) + p20, (-self.l2 * ca.cos(q[3])) + p21
        p50, p51 = (self.l1 * ca.sin(q[4])) + p40, (-self.l1 * ca.cos(q[4])) + p41

        dp10, dp11 = ca.jtimes(p10, ca.MX(q), dq), ca.jtimes(p11, ca.MX(q), dq)
        dp20, dp21 = ca.jtimes(p20, ca.MX(q), dq), ca.jtimes(p21, ca.MX(q), dq)
        dp30, dp31 = ca.jtimes(p30, ca.MX(q), dq), ca.jtimes(p31, ca.MX(q), dq)
        dp40, dp41 = ca.jtimes(p40, ca.MX(q), dq), ca.jtimes(p41, ca.MX(q), dq)
        dp50, dp51 = ca.jtimes(p50, ca.MX(q), dq), ca.jtimes(p51, ca.MX(q), dq)
        p = [[p10, p11], [p20, p21], [p30, p31], [p40, p41], [p50, p51]]
        dp = [[dp10, dp11], [dp20, dp21], [dp30, dp31], [dp40, dp41], [dp50, dp51]]

        ##########################################
        c10, c11 = -self.l1 * ca.sin(q[0]) / 2 + p0[0], self.l1 * ca.cos(q[0]) / 2 + p0[0]
        c20, c21 = -self.l2 * ca.sin(q[1]) / 2 + p10, self.l2 * ca.cos(q[1]) / 2 + p11
        c30, c31 = self.l3 * ca.sin(q[2]) / 2 + p20, self.l3 * ca.cos(q[2]) / 2 + p21
        c40, c41 = (self.l2 * ca.sin(q[3]) / 2) + p20, (-self.l2 * ca.cos(q[3]) / 2) + p21
        c50, c51 = (self.l1 * ca.sin(q[4]) / 2) + p40, (-self.l1 * ca.cos(q[4]) / 2) + p41
        # print(c1)
        # print(c2)
        dc10, dc11 = ca.jtimes(c10, ca.MX(q), dq), ca.jtimes(c11, ca.MX(q), dq)
        dc20, dc21 = ca.jtimes(c20, ca.MX(q), dq), ca.jtimes(c21, ca.MX(q), dq)
        dc30, dc31 = ca.jtimes(c30, ca.MX(q), dq), ca.jtimes(c31, ca.MX(q), dq)
        dc40, dc41 = ca.jtimes(c40, ca.MX(q), dq), ca.jtimes(c41, ca.MX(q), dq)
        dc50, dc51 = ca.jtimes(c50, ca.MX(q), dq), ca.jtimes(c51, ca.MX(q), dq)
        g = [[c10, c11], [c20, c21], [c30, c31], [c40, c41], [c50, c51]]
        dc = [[dc10, dc11], [dc20, dc21], [dc30, dc31], [dc40, dc41], [dc50, dc51]]
        # print(dc1)
        return p, dp, g, dc

    def heelStrike(self, q, dq, p, dp, g, dg):
        qi = [q[4], q[3], q[2], q[1], q[0]]
        pi = [p[4], p[3], p[2], p[1], p[0]]
        dpi = [dp[4], dp[3], dp[2], dp[1], dp[0]]
        gi = [g[4], g[3], g[2], g[1], g[0]]
        dgi = [dg[4], dg[3], dg[2], dg[1], dg[0]]
        s = [0, 0, 0, 0, 0]
        for i in range(5):
            s[0] += ((self.m[i] * (((g[i][0] - p[4][0]) * dg[i][1]) - ((g[i][1] - p[4][1]) * dg[i][0]))) + (
                        self.i[i] * dq[i])
                     - (self.m[i] * (((gi[i][0] - pi[0][0]) * dgi[i][1]) - ((gi[i][1] - pi[0][1]) * dgi[i][0]))))
            if i < 4:
                s[1] += ((self.m[i] * (((g[i][0] - p[3][0]) * dg[i][1]) - ((g[i][1] - p[3][1]) * dg[i][0]))) + (
                            self.i[i] * dq[i])
                         - (self.m[i] * (((gi[i][0] - pi[1][0]) * dgi[i][1]) - ((gi[i][1] - pi[1][1]) * dgi[i][0]))))
                if i < 3:
                    s[2] += ((self.m[i] * (((g[i][0] - p[1][0]) * dg[i][1]) - ((g[i][1] - p[1][1]) * dg[i][0]))) + (
                                self.i[i] * dq[i])
                             - (self.m[i] * (
                                        ((gi[i][0] - pi[1][0]) * dgi[i][1]) - ((gi[i][1] - pi[1][1]) * dgi[i][0]))))
                    if i < 2:
                        s[3] += ((self.m[i] * (((g[i][0] - p[1][0]) * dg[i][1]) - ((g[i][1] - p[1][1]) * dg[i][0]))) + (
                                    self.i[i] * dq[i])
                                 - (self.m[i] * (
                                            ((gi[i][0] - pi[1][0]) * dgi[i][1]) - ((gi[i][1] - pi[1][1]) * dgi[i][0]))))
                        if i < 1:
                            s[4] += ((self.m[i] * (
                                        ((g[i][0] - p[0][0]) * dg[i][1]) - ((g[i][1] - p[0][1]) * dg[i][0]))) + (
                                                 self.i[i] * dq[i])
                                     - (self.m[i] * (((gi[i][0] - pi[3][0]) * dgi[i][1]) - (
                                                (gi[i][1] - pi[3][1]) * dgi[i][0]))))
        dqi = [0, 0, 0, 0, 0]
        dqi[4] = s[4] / self.i[4]
        dqi[3] = s[3] - s[4] / self.i[3]
        dqi[2] = s[2] - s[3] / self.i[2]
        dqi[1] = s[1] - s[2] / self.i[1]
        dqi[0] = s[0] - s[1] / self.i[0]

        return [qi, dqi]


class nlp(walker):
    def __init__(self, walker):
        self.cost = self.getCost(walker.u, walker.N, walker.h)
        walker.opti.minimize(self.cost)
        self.ceq = self.getConstraints(walker)
        walker.opti.subject_to(self.ceq)
        self.bounds = self.getBounds(walker)
        walker.opti.subject_to(self.bounds)
        p_opts = {"expand": True}
        s_opts = {"max_iter": 1000}
        walker.opti.solver("ipopt", p_opts, s_opts)
        self.initial = self.initalGuess(walker)

    def initalGuess(self, walker):
        iniq = np.zeros((5, walker.N))
        inidq = np.zeros((5, walker.N))
        iniu = np.zeros((4, walker.N))
        for j in range(5):
            for i in range(walker.N):
                walker.opti.set_initial(walker.state[i][0][j],
                                        (walker.goal[0][j] + (i / (walker.N - 1)) * (
                                                    walker.goal[1][j] - walker.goal[0][j])))
                iniq[j, i] = (walker.goal[0][j] + (i / (walker.N - 1)) * (walker.goal[1][j] - walker.goal[0][j]))

                walker.opti.set_initial(walker.state[i][1][j],
                                        (walker.goal[1][j] - walker.goal[0][j]) / (walker.N - 1))
                inidq[j, i] = (walker.goal[1][j] - walker.goal[0][j]) / (walker.N - 1)

                if j < 4:
                    walker.opti.set_initial(walker.u[i][0][j], 0)

        return [iniq, inidq, iniu]

    def getCost(self, u, N, h):
        result = 0
        for i in range(N - 1):
            for j in range(4):
                result += (h / 2) * (u[i][0][j] ** 2 + u[i + 1][0][j] ** 2)
        return result

    def getConstraints(self, walker):
        ceq = []
        for i in range(walker.N - 2):
            q1 = (walker.state[i][0])
            q2 = (walker.state[i + 1][0])
            dq1 = (walker.state[i][1])
            dq2 = (walker.state[i + 1][1])
            dq12 = (walker.halfstate[i][0])
            ddq1 = walker.ddq[i]
            ddq2 = walker.ddq[i + 1]
            ddq12 = walker.ddq12[i]
            ceq.extend(self.getCollocation1(q1, q2,
                                           dq1, dq2, dq12, ddq1, ddq2, ddq12,
                                           walker.h))

        q0 = (walker.state[0][0])
        dq0 = (walker.state[0][1])
        qf = (walker.state[-1][0])
        ceq.extend(self.getBoundaryConstrainsts(q0, dq0, qf, walker.goal, walker.impactmap))
        #ceq.extend(self.getBoundaryConstrainsts(q0, dq0, qf, walker.goal))
        ceq.extend([(walker.dp0[4][1] >= 0), (walker.dpN[4][1] <= 0)])
        for i in range(walker.N):
            ceq.extend([((walker.pos[i][4][0]) <= walker.step_max)])
            ceq.extend([((walker.pos[i][4][0]) >= -walker.step_max)])
            ceq.extend([((walker.pos[i][4][1]) >= 0)])
            ceq.extend([((walker.pos[i][4][1]) <= 0.2)])
        #ceq.extend([((walker.pos[-2][4][1])==0)])
        ceq.extend([((walker.pos[-1][4][1]) == 0)])
        return ceq

    def getCollocation(self, q1, q2, dq1, dq2, dq12, ddq1, ddq2, ddq12, h):
        cc = []

        for i in range(4):
            #dq12 = dq1 + (h / 2) * ddq1
            #cc.extend([(dq12 - dq1 - (h / 2) * ddq1==0)])
            cc.extend([(((h / 2) * (ddq2[i] + ddq1[i])) - (dq2[i] - dq1[i]) == 0)])
            cc.extend([(((h / 2) * (dq2[i] + dq1[i])) - (q2[i] - q1[i]) == 0)])
        return cc

    def getCollocation1(self, q1, q2, dq1, dq2, dq12, ddq1, ddq2, ddq12, h):
        cc=[]
        for i in range(4):
            cc.extend([((dq12[i]-dq1[i]-(h/2)*ddq1[i])==0)])
            cc.extend([((q2[i]-q1[i]-h*dq12[i])==0)])
            cc.extend([((dq2[i]-dq12[i]-(h/2)*ddq12[i])==0)])
        return cc



    def getBoundaryConstrainsts(self, state1, dstate1, state2, goal, impact):
        c = []
        for i in range(4): c.extend([(state1[i] - impact[0][i] == 0), (dstate1[i] - impact[1][i] == 0)
                                        , ((state2[i] - goal[1][i]) == 0)])
        # for i in range(4): c.extend([(state1[i] - impact[0][i] == 0)
        #                                 , ((state2[i] - goal[1][i]) == 0)])
        #for i in range(4): c.extend([((state1[i] - goal[0][i]) == 0), ((state2[i] - goal[1][i]) == 0)])
        #for i in range(4): c.extend([((state1[i] - goal[0][i]) == 0), ((state2[i] - goal[1][i]) == 0)])
        return c

    def getBounds(self, walker):
        c = []
        f = 3.5
        for i in range(walker.N):
            q = (walker.state[i][0])
            dq = (walker.state[i][1])
            u = (walker.u[i][0])
            c.extend([walker.opti.bounded(-walker.pi, q[0], walker.pi),
                      walker.opti.bounded(-walker.pi, q[1], walker.pi),
                      walker.opti.bounded(-walker.pi, q[2], walker.pi),
                      walker.opti.bounded(-walker.pi, q[3], walker.pi),
                      walker.opti.bounded(-walker.pi, q[4], walker.pi),
                      walker.opti.bounded(-f * walker.pi, dq[0], f * walker.pi),
                      walker.opti.bounded(-f * walker.pi, dq[1], f * walker.pi),
                      walker.opti.bounded(-f * walker.pi, dq[2], f * walker.pi),
                      walker.opti.bounded(-f * walker.pi, dq[3], f * walker.pi),
                      walker.opti.bounded(-f * walker.pi, dq[4], f * walker.pi),
                      # walker.opti.bounded(0,u[0],0),
                      walker.opti.bounded(-walker.tauMax, u[0], walker.tauMax),
                      walker.opti.bounded(-walker.tauMax, u[1], walker.tauMax),
                      walker.opti.bounded(-walker.tauMax, u[2], walker.tauMax),
                      walker.opti.bounded(-walker.tauMax, u[3], walker.tauMax)])
        return c


me = walker()
n1 = nlp(me)

sol1 = me.opti.solve()

q = [];
dq = [];
u = [];
pos = []

for j in range(5):
    tempq = [];
    tempdq = [];
    tempu = [];
    temp = []
    for i in range(me.N):
        tempq.append(sol1.value(me.state[i][0][j]))
        tempdq.append(sol1.value(me.state[i][1][j]))
        temp.append([sol1.value(me.pos[i][j][0]), sol1.value(me.pos[i][j][1])])
        if j < 4:
            tempu.append(sol1.value(me.u[i][0][j]))
    q.append(tempq);
    pos.append(temp)
    dq.append(tempdq)
    u.append(tempu)
time = np.arange(0.0, me.T, me.h)
from matplotlib import animation
from celluloid import Camera

fig = plt.figure()
camera = Camera(fig)
for i in range(me.N):
    p1 = [pos[0][i][1], pos[0][i][0]]
    p2 = [pos[1][i][1], pos[1][i][0]]
    p3 = [pos[2][i][1], pos[2][i][0]]
    p4 = [pos[3][i][1], pos[3][i][0]]
    p5 = [pos[4][i][1], pos[4][i][0]]
    # plt.axes(xlim=(-2, 2), ylim=(-2, 2))
    plt.axes(xlim=(-2, 2), ylim=(-2, 2))
    # plt.plot([0,-p1[1]], [0,p1[0]],'r',[-p1[1],-p2[1]], [p1[0],p2[0]],'b',
    #     [-p2[1],-p3[1]], [p2[0],p3[0]],'c',
    #     [-p2[1],p4[1] - 2*p2[1]], [p2[0],2*p2[0]-p4[0]],'b',
    #     [p4[1] - 2*p2[1],p5[1]], [2*p2[0]-p4[0],(p5[0] - 2*p2[0])],'r')
    plt.plot([0, p1[1]], [0, p1[0]], 'r', [p1[1], p2[1]], [p1[0], p2[0]], 'g',
             [p2[1], p3[1]], [p2[0], p3[0]], 'b', [p2[1], p4[1]], [p2[0], p4[0]], 'y',
             [p4[1], p5[1]], [p4[0], p5[0]], 'c')

    plt.plot([-2, 2], [0, 0], 'g')
    # if cv2.waitKey(0) & 0xFF == ord("q"):
    # #     break
    camera.snap()
animation = camera.animate(interval=1, repeat= False)
animation.save('animation3.gif')
plt.show()
plt.close()

name = ['q', 'dq', 'u']

plt.subplot(322)
plt.title('Optimised Solution')
plt.plot(time, q[:][0], 'r', time, q[:][1], 'g', time, q[:][2], 'b',
         time, q[:][3], 'y', time, q[:][4], 'c')

plt.subplot(321)
plt.title('Initial Guess')
iniq = n1.initial[0]
plt.plot(time, iniq[:][0], 'r', time, iniq[:][1], 'g', time, iniq[:][2], 'b',
         time, iniq[:][3], 'y', time, iniq[:][4], 'c')
plt.ylabel(name[0])

plt.subplot(324)
plt.plot(time, dq[:][0], 'r', time, dq[:][1], 'g', time, dq[:][2], 'b',
         time, dq[:][3], 'y', time, dq[:][4], 'c')

plt.subplot(323)
inidq = n1.initial[1]
plt.plot(time, inidq[:][0], 'r', time, inidq[:][1], 'g', time, inidq[:][2], 'b',
         time, inidq[:][3], 'y', time, inidq[:][4], 'c')
plt.ylabel(name[1])

plt.subplot(326)
plt.plot(time, u[:][0], 'g', time, u[:][1], 'b', time, u[:][2], 'y',
         time, u[:][3], 'c')

plt.subplot(325)
iniu = n1.initial[2]
plt.plot(time, iniu[:][0], 'r', time, iniu[:][1], 'g', time, iniu[:][2], 'b',
         time, iniu[:][3], 'y')
plt.ylabel(name[2])

plt.suptitle('Five-Link')
plt.show()
plt.savefig('five_link')

#### qi
plt.title('Initial Guess')
iniq = n1.initial[0]
plt.plot(time, iniq[:][0], 'r', label='q1')
plt.plot(time, iniq[:][1], 'g', label='q2')
plt.plot(time, iniq[:][2], 'b', label='q3')
plt.plot(time, iniq[:][3], 'y', label='q4')
plt.plot(time, iniq[:][4], 'c', label='q5')
plt.ylabel('q')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.savefig('qi')
plt.show()

### dqi
plt.title('Initial Guess')
iniq = n1.initial[1]
plt.plot(time, iniq[:][0], 'r', label='dq1')
plt.plot(time, iniq[:][1], 'g', label='dq2')
plt.plot(time, iniq[:][2], 'b', label='dq3')
plt.plot(time, iniq[:][3], 'y', label='dq4')
plt.plot(time, iniq[:][4], 'c', label='dq5')
plt.ylabel('dq')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.savefig('dqi')
plt.show()


#### ui
plt.title('Initial Guess')
iniq = n1.initial[2]
plt.plot(time, iniq[:][0], 'g', label='u2')
plt.plot(time, iniq[:][1], 'b', label='u3')
plt.plot(time, iniq[:][2], 'y', label='u4')
plt.plot(time, iniq[:][3], 'c', label='u5')
plt.ylabel('u')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.savefig('ui')
plt.show()


##### qf
plt.title('Optimised Solution')
iniq = n1.initial[0]
plt.plot(time, q[:][0], 'r', label='q1')
plt.plot(time, q[:][1], 'g', label='q2')
plt.plot(time, q[:][2], 'b', label='q3')
plt.plot(time, q[:][3], 'y', label='q4')
plt.plot(time, q[:][4], 'c', label='q5')
plt.ylabel('q')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.savefig('qf')
plt.show()

### dqf
plt.title('Optimised Solution')
iniq = n1.initial[1]
plt.plot(time, dq[:][0], 'r', label='dq1')
plt.plot(time, dq[:][1], 'g', label='dq2')
plt.plot(time, dq[:][2], 'b', label='dq3')
plt.plot(time, dq[:][3], 'y', label='dq4')
plt.plot(time, dq[:][4], 'c', label='dq5')
plt.ylabel('dq')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.savefig('dqf')
plt.show()


#### uf
plt.title('Optimised Solution')
iniq = n1.initial[2]
plt.plot(time, u[:][0], 'g', label='u2')
plt.plot(time, u[:][1], 'b', label='u3')
plt.plot(time, u[:][2], 'y', label='u4')
plt.plot(time, u[:][3], 'c', label='u5')
plt.ylabel('u')
plt.xlabel('t')
plt.legend(loc='lower right')
plt.savefig('uf')
plt.show()




i=99
p1 = [pos[0][i][1], pos[0][i][0]]
p2 = [pos[1][i][1], pos[1][i][0]]
p3 = [pos[2][i][1], pos[2][i][0]]
p4 = [pos[3][i][1], pos[3][i][0]]
p5 = [pos[4][i][1], pos[4][i][0]]
# plt.axes(xlim=(-2, 2), ylim=(-2, 2))
plt.axes(xlim=(-2, 2), ylim=(-2, 2))
# plt.plot([0,-p1[1]], [0,p1[0]],'r',[-p1[1],-p2[1]], [p1[0],p2[0]],'b',
#     [-p2[1],-p3[1]], [p2[0],p3[0]],'c',
#     [-p2[1],p4[1] - 2*p2[1]], [p2[0],2*p2[0]-p4[0]],'b',
#     [p4[1] - 2*p2[1],p5[1]], [2*p2[0]-p4[0],(p5[0] - 2*p2[0])],'r')
plt.plot([0, p1[1]], [0, p1[0]], 'r', [p1[1], p2[1]], [p1[0], p2[0]], 'r',
         [p2[1], p3[1]], [p2[0], p3[0]], 'r', [p2[1], p4[1]], [p2[0], p4[0]], 'r',
         [p4[1], p5[1]], [p4[0], p5[0]], 'r')

plt.plot([-2, 2], [0, 0], 'g')
plt.title('t=0.1')
plt.savefig('walking6')
plt.show()
