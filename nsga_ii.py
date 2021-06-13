import numpy as np
#import pandas as pd
import random
import math
import time
#from numba import njit
#from multiprocessing import Process
#import concurrent.futures

# SETS
visit = 50
periods = 5
patients = 50
vehicles = 5
pharmacies = 6
rs = 3
labs = 2
drug = 5
numcol_s = vehicles * (2 + patients + rs) + vehicles - 1
# PARAMETERES
we = 0.4
wei = 0.6
cv = []
for i in range(vehicles):
    cv.append(round(random.uniform(0.36, 0.70), 3))
rv = []
for i in range(vehicles):
    rv.append(round(random.uniform(1.2, 4.2), 3))

t = np.zeros((pharmacies+patients+rs+labs,pharmacies+patients+rs+labs,periods))
for i in range(t.shape[0]):
    for j in range(t.shape[1]):
        for k in range(t.shape[2]):
            t[i,j,k] = round(random.uniform(10, 40))


sch =[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15., 16.,  0., 18., 19.,  0., 21.,  0.,  0.,  0.,
  25.,  0.,  0.,  0., 29.,  0., 31., 32.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  9.,  0., 11., 12., 13.,  0.,  0., 16., 17.,  0., 19., 20., 21.,  0.,  0., 24.,
   0., 26.,  0.,  0., 29., 30.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  8.,  0.,  0.,  0., 12., 13.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 23.,  0.,
   0., 26.,  0.,  0., 29.,  0., 31.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  0.,  0., 10.,  0.,  0.,  0., 14.,  0.,  0.,  0., 18.,  0., 20., 21.,  0.,  0., 24.,
   0., 26.,  0., 28.,  0., 30.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
 [ 0.,  8.,  0., 10. , 0.,  0.,  0., 14.,  0. , 0. , 0. ,18.,  0., 20., 21.,  0.,  0., 24.,
  25., 26., 27., 28. , 0. , 0., 31.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]

sch = np.asmatrix(sch)
print(sch)


TC = []
for i in range(vehicles):
    TC.append((random.uniform(1.5, 2.5)))
CC = []
for i in range(vehicles):
    CC.append((random.uniform(1.86, 3.54)))

OCP = []
for i in range(pharmacies):
    OCP.append((random.uniform(600000, 1300000)))

JO = []
for i in range(pharmacies):
    JO.append(round(random.uniform(5, 15)))
print(JO)
UR = []
for i in range(pharmacies):
    UR.append(round(random.uniform(0.08, 0.14), 3))
print(UR)
ev = []
for i in range(pharmacies):
    ev.append(round(random.uniform(0.1, 1.7), 3))
print(ev)
rd = []
for i in range(pharmacies):
    rd.append(round(random.uniform(0.04, 0.93), 3))
print(rd)

A = np.zeros((patients,drug))
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        A[i,j] = round(random.uniform(5, 80))
A = np.asmatrix(A)

CAP = np.zeros((pharmacies,drug,periods))
for i in range(CAP.shape[0]):
    for j in range(CAP.shape[1]):
        for k in range(CAP.shape[2]):
            CAP[i,j,k] = round(random.uniform(800, 2000))


q = []
for i in range(patients):
    q.append(round(random.uniform(1, 4)))
print(q)

qp = [4, 4, 5, 4, 5]

print(qp)

E = np.zeros((periods,patients))
for i in range(E.shape[0]):
    for j in range(E.shape[1]):
        E[i,j] = round(random.uniform(1, 10),4)
E = np.asmatrix(E)
#print(E)

L = np.zeros((periods,patients))
for i in range(L.shape[0]):
    for j in range(L.shape[1]):
        L[i,j] = round(random.uniform(800, 900))
L = np.asmatrix(L)
#print(L)

DV = []
for i in range(pharmacies):
    DV.append(int(0))
for i in range(patients):
    DV.append(round(random.uniform(5, 30),2))
#print(DV)

rstation = []
for j7 in range(pharmacies + patients + 1, pharmacies + patients + rs + 1):
    rstation.append(j7)

pharm = []
for j in range(1, pharmacies + 1):
    pharm.append(j)
###
firstz2 = []
secondz2 = []

for jj in range(pharmacies):
    firstz2.append(JO[jj] * UR[jj])
    secondz2.append(ev[jj] * (1 - rd[jj]))
firstz2min = min(firstz2)
firstz2msax = sum(firstz2)
secondz2min = min(secondz2)
secondz2max = sum(secondz2)
print(firstz2)
print(secondz2)
print(firstz2min)
print(firstz2msax)
print(secondz2min)
print(secondz2max)
###
def index_min(dic,minn):
    for name, age in dic.items():
        if age == minn:
            return name

# GENERATING RANDOM SOLUTION
#@jit(nopython=True)
def generate_solution(periods, patients, rs, pharmacies, labs, vehicles, cv, t, sch,rstation):
    s = np.zeros((periods * 2, (vehicles * (2 + patients + rs)) + (vehicles - 1)))
    h = np.zeros((periods, (vehicles * (2 + patients + rs)) + (vehicles - 1)))
    ss = np.zeros((periods, (vehicles * (2 + patients + rs)) + (vehicles - 1)))
    CD = np.zeros((periods, (vehicles * (2 + patients + rs)) + (vehicles - 1)))
    for i in range(s.shape[0]):
        if i % 2 == 0:
            it1 = 0
            it2 = 1
            it3 = 1
            for j in range(s.shape[1]):
                if j == (it1 * ((2 + patients + rs) + 1)):
                    s[i, j] = random.randint(1, pharmacies)
                    it1 = it1 + 1
                elif j == (it2 * ((2 + patients + rs) + 1)) - 2:
                    TimeToLab = {k: 0 for k in range(pharmacies + patients + rs + 1, pharmacies + patients + rs + labs + 1)}
                    for ii in TimeToLab.keys():
                        TimeToLab[ii] = t[int(s[i, j - 1]) - 1, int(ii) - 1, int(i / 2)]
                    minnn = min(i for i in TimeToLab.values() if i != 0)
                    s[i, j] = index_min(TimeToLab, minnn)
                    it2 = it2 + 1
                elif j == (it3 * ((2 + patients + rs) + 1)) - 1:
                    s[i, j] = 0
                    it3 = it3 + 1
                else:
                    TimeTo = {k: 0 for k in (sch[int(i / 2), :]).tolist()[0]}
                    if len(TimeTo.keys()) == 1:
                        continue
                    else:
                        for ii in TimeTo.keys():
                            TimeTo[ii] = t[int(s[i, j-1]) - 1, int(ii) - 1, int(i / 2)]
                    #print(TimeTo)
                        minn = min(i for i in TimeTo.values() if i != 0 and index_min(TimeTo,i) != 0)
                    #print(minn)
                        s[i, j] = index_min(TimeTo,minn)


            for j1 in range(s.shape[1]):
                for j2 in range(j1):
                    if s[i, j1] != 0 and s[i, j2] != 0:
                        if s[i, j1] not in list(
                                range(pharmacies + patients + rs + 1, pharmacies + patients + rs + labs + 1)) and s[
                            i, j2] not in list(
                            range(pharmacies + patients + rs + 1, pharmacies + patients + rs + labs + 1)):
                            if s[i, j1] not in list(range(1, pharmacies + 1)) and s[i, j2] not in list(
                                    range(1, pharmacies + 1)):
                                if s[i, j1] == s[i, j2]:
                                    for n in (sch[int(i / 2), :]).tolist()[0]:
                                        if n != 0:
                                            if n not in s[i, :]:
                                                s[i, j1] = n
                                                break

                                            else:
                                                s[i, j1] = 999
            #print(s)

        else:
            it4 = 0
            it5 = 1
            it6 = 1
            it8 = 0
            for j in range(s.shape[1]):
                if j == (it4 * (2 + patients + rs + 1)):
                    s[i, j] = 100
                    ss[int((i - 1) / 2), j] = 0
                    it4 = it4 + 1
                elif j == (it6 * (2 + patients + rs + 1)) - 1:
                    s[i, j] = 0
                    ss[int((i - 1) / 2), j] = 0
                    it6 = it6 + 1
                elif (it8 * (2 + patients + rs + 1) + 1) <= j <= ((it8 + 1) * (2 + patients + rs + 1) - 2):
                    if s[i - 1, j] == 999:
                        s[i, j] = 0
                        ss[int((i - 1) / 2), j] = 0

                    else:
                        if j == it8 * (patients + rs + 2 + 1) + 1:
                            pre3 = it8 * (patients + rs + 2 + 1)
                        else:
                            for j5 in range(j - 1, it8 * (patients + rs + 2 + 1) - 1, -1):
                                if s[i - 1, j5] != 999:
                                    pre3 = j5
                                    break

                        s[i, j] = s[i, pre3] - cv[it8] * t[
                            int(s[i - 1, pre3]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)]
                        if s[i - 1, pre3] in rstation:
                            ss[int((i - 1) / 2), j] = ss[int((i - 1) / 2), pre3] + t[
                                int(s[i - 1, pre3]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)] + CD[
                                                          int((i - 1) / 2), pre3]
                        else:
                            ss[int((i - 1) / 2), j] = ss[int((i - 1) / 2), pre3] + t[
                                int(s[i - 1, pre3]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)] + DV[
                                                          int(s[i - 1, pre3]) - 1]

                        if s[i, j] < 0:

                            TimeToRS = []
                            for iii in range(pharmacies + patients + 1, pharmacies + patients + rs + 1):
                                TimeToRS.append(t[int(s[i - 1, pre3]) - 1, int(iii) - 1, int((i - 1) / 2)])

                            station = TimeToRS.index(min(TimeToRS)) + (pharmacies + patients + 1)
                            last = s[i - 1, j]
                            s[i - 1, j] = station
                            h[int((i - 1) / 2), j] = s[i, pre3] - cv[it8] * t[
                                int(s[i - 1, pre3]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)]
                            s[i, j] = random.randint(90, 100)
                            CD[int((i - 1) / 2), j] = rv[it8] * (s[i, j] - h[int((i - 1) / 2), j])
                            ss[int((i - 1) / 2), j] = ss[int((i - 1) / 2), pre3] + t[
                                int(s[i - 1, pre3]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)] + DV[
                                                          int(s[i - 1, pre3]) - 1]

                            if j == (it8 + 1) * (2 + patients + rs + 1) - 2:
                                for j3 in range(j - 1, 0, -1):
                                    if s[i - 1, j3] == 999:
                                        pre5 = j3
                                        break
                                ad = 0
                                for j4 in range(pre5 + 1, (it8 + 1) * (2 + patients + rs + 1) - 1):
                                    s[i - 1, pre5 + ad] = s[i - 1, j4]
                                    s[i, pre5 + ad] = s[i, j4]
                                    ss[int((i - 1) / 2), pre5 + ad] = ss[int((i - 1) / 2), j4]
                                    ad = ad + 1
                                s[i - 1, j] = last
                                s[i, j] = s[i, j - 1] - cv[it8] * t[
                                    int(s[i - 1, j - 1]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)]
                                ss[int((i - 1) / 2), j] = ss[int((i - 1) / 2), j - 1] + t[
                                    int(s[i - 1, j - 1]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)] + CD[
                                                              int((i - 1) / 2), j - 1]
                            else:
                                llast = s[i - 1, j + 1]
                                s[i - 1, j + 1] = last
                                s[i, j + 1] = s[i, j] - cv[it8] * t[
                                    int(s[i - 1, j]) - 1, int(s[i - 1, j + 1]) - 1, int((i - 1) / 2)]
                                ss[int((i - 1) / 2), j + 1] = ss[int((i - 1) / 2), j] + t[
                                    int(s[i - 1, j]) - 1, int(s[i - 1, j + 1]) - 1, int((i - 1) / 2)] + CD[
                                                                  int((i - 1) / 2), j]

                                if j + 1 == (it8 + 1) * (2 + patients + rs + 1) - 2:
                                    for j3 in range(j, 0, -1):
                                        if s[i - 1, j3] == 999:
                                            pre2 = j3
                                            break
                                    ad = 0
                                    for j4 in range(pre2 + 1, (it8 + 1) * (2 + patients + rs + 1) - 1):
                                        s[i - 1, pre2 + ad] = s[i - 1, j4]
                                        s[i, pre2 + ad] = s[i, j4]
                                        ss[int((i - 1) / 2), pre2 + ad] = ss[int((i - 1) / 2), j4]
                                        ad = ad + 1
                                    s[i - 1, j + 1] = llast
                                    s[i, j + 1] = s[i, j] - cv[it8] * t[
                                        int(s[i - 1, j]) - 1, int(s[i - 1, j + 1]) - 1, int((i - 1) / 2)]
                                    ss[int((i - 1) / 2), j + 1] = ss[int((i - 1) / 2), j] + t[
                                        int(s[i - 1, j]) - 1, int(s[i - 1, j + 1]) - 1, int((i - 1) / 2)] + DV[
                                                                      int(s[i - 1, j]) - 1]

                    if j == ((it8 + 1) * (2 + patients + rs + 1) - 2):
                        it8 = it8 + 1

    #print(s)
    return (s, ss, h)

def adjust_solution(s,periods, patients, rs, pharmacies, labs, vehicles, cv, t, sch,rstation):
    h = np.zeros((periods, (vehicles * (2 + patients + rs)) + (vehicles - 1)))
    ss = np.zeros((periods, (vehicles * (2 + patients + rs)) + (vehicles - 1)))
    CD = np.zeros((periods, (vehicles * (2 + patients + rs)) + (vehicles - 1)))
    for i in range(s.shape[0]):
        if i % 2 == 0:
            it1 = 0
            it2 = 1
            it3 = 1
            for j in range(s.shape[1]):
                if j == (it2 * ((2 + patients + rs) + 1)) - 2:
                    TimeToLab = {k: 0 for k in range(pharmacies + patients + rs + 1, pharmacies + patients + rs + labs + 1)}
                    for j5 in range(j - 1, (it2-1) * (patients + rs + 2 + 1) - 1, -1):
                        if s[i , j5] != 999:
                            pre33 = j5
                            break
                    if pre33 == ((it2-1) * ((2 + patients + rs) + 1)):
                        s[i, j]= pharmacies + patients + rs + 1
                    else:
                        for ii in TimeToLab.keys():
                            TimeToLab[ii] = t[int(s[i, pre33]) - 1, int(ii) - 1, int(i / 2)]
                        #print(TimeToLab)
                        minnn = min(i for i in TimeToLab.values() if i != 0)
                        s[i, j] = index_min(TimeToLab, minnn)
                        it2 = it2 + 1

            for j1 in range(s.shape[1]):
                for j2 in range(j1):
                    if s[i, j1] != 0 and s[i, j2] != 0:
                        if s[i, j1] not in list(
                                range(pharmacies + patients + rs + 1, pharmacies + patients + rs + labs + 1)) and s[
                            i, j2] not in list(
                            range(pharmacies + patients + rs + 1, pharmacies + patients + rs + labs + 1)):
                            if s[i, j1] not in list(range(1, pharmacies + 1)) and s[i, j2] not in list(
                                    range(1, pharmacies + 1)):
                                if s[i, j1] == s[i, j2]:
                                    for n in (sch[int(i / 2), :]).tolist()[0]:
                                        if n != 0:
                                            if n not in s[i, :]:
                                                s[i, j1] = n
                                                break

                                            else:
                                                s[i, j1] = 999
            it12 = 0
            for j in range(s.shape[1]):
                if s[i, j] in rstation:
                    for jj in range(j - 1, it12 * (patients + rs + 2 + 1) - 1, -1):
                        if s[i - 1, jj] != 999:
                            pre6 = jj
                            break
                    for jjj in range(j + 1, ((it12 + 1) * (2 + patients + rs + 1) - 1)):
                        if s[i, jjj] != 999:
                            nex = jjj
                            break

                    if s[i, nex] == s[i, j] or s[i, nex] == s[i, pre6]:
                        s[i, nex] = 999
                    if s[i, nex] == s[i, j] and s[i, nex] == s[i, pre6]:
                        s[i, j] = 999
                        s[i, nex] = 999

                if j == ((it12 + 1) * (2 + patients + rs + 1) - 2):
                    it12 = it12 + 1

            count = 0
            for j in range(s.shape[1]):
                if s[i,j] in rstation:
                    count = count + 1
                    if count >= 2:
                        s[i,j] = 999

            for n in (sch[int(i / 2), :]).tolist()[0]:
                if n != 0:
                    if n not in s[i, :]:
                        for j8 in range(s.shape[1]):
                            if s[i, j8] == 999:
                                s[i, j8] = n

                                # print()

        else:
            it4 = 0
            it5 = 1
            it6 = 1
            it8 = 0
            for j in range(s.shape[1]):
                if j == (it4 * (2 + patients + rs + 1)):
                    s[i, j] = 100
                    ss[int((i - 1) / 2), j] = 0
                    it4 = it4 + 1
                elif j == (it6 * (2 + patients + rs + 1)) - 1:
                    s[i, j] = 0
                    ss[int((i - 1) / 2), j] = 0
                    it6 = it6 + 1
                elif (it8 * (2 + patients + rs + 1) + 1) <= j <= ((it8 + 1) * (2 + patients + rs + 1) - 2):
                    if s[i - 1, j] == 999:
                        s[i, j] = 0
                        ss[int((i - 1) / 2), j] = 0

                    elif s[i - 1, j] in rstation:
                        s[i, j] = random.randint(90, 100)

                    else:
                        if j == it8 * (patients + rs + 2 + 1) + 1:
                            pre3 = it8 * (patients + rs + 2 + 1)
                        else:
                            for j5 in range(j - 1, it8 * (patients + rs + 2 + 1) - 1, -1):
                                if s[i - 1, j5] != 999:
                                    pre3 = j5
                                    break

                        s[i, j] = s[i, pre3] - cv[it8] * t[
                            int(s[i - 1, pre3]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)]
                        if s[i - 1, pre3] in rstation:
                            ss[int((i - 1) / 2), j] = ss[int((i - 1) / 2), pre3] + t[
                                int(s[i - 1, pre3]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)] + CD[
                                                          int((i - 1) / 2), pre3]
                        else:
                            ss[int((i - 1) / 2), j] = ss[int((i - 1) / 2), pre3] + t[
                                int(s[i - 1, pre3]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)] + DV[
                                                          int(s[i - 1, pre3]) - 1]

                        if s[i, j] < 0:

                            TimeToRS = []
                            for iii in range(pharmacies + patients + 1, pharmacies + patients + rs + 1):
                                TimeToRS.append(t[int(s[i - 1, pre3]) - 1, int(iii) - 1, int((i - 1) / 2)])

                            station = TimeToRS.index(min(TimeToRS)) + (pharmacies + patients + 1)
                            last = s[i - 1, j]
                            s[i - 1, j] = station
                            h[int((i - 1) / 2), j] = s[i, pre3] - cv[it8] * t[
                                int(s[i - 1, pre3]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)]
                            s[i, j] = random.randint(90, 100)
                            CD[int((i - 1) / 2), j] = rv[it8] * (s[i, j] - h[int((i - 1) / 2), j])
                            ss[int((i - 1) / 2), j] = ss[int((i - 1) / 2), pre3] + t[
                                int(s[i - 1, pre3]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)] + DV[
                                                          int(s[i - 1, pre3]) - 1]

                            if j == (it8 + 1) * (2 + patients + rs + 1) - 2:
                                for j3 in range(j - 1, 0, -1):
                                    if s[i - 1, j3] == 999:
                                        pre5 = j3
                                        break
                                ad = 0
                                for j4 in range(pre5 + 1, (it8 + 1) * (2 + patients + rs + 1) - 1):
                                    s[i - 1, pre5 + ad] = s[i - 1, j4]
                                    s[i, pre5 + ad] = s[i, j4]
                                    ss[int((i - 1) / 2), pre5 + ad] = ss[int((i - 1) / 2), j4]
                                    ad = ad + 1
                                s[i - 1, j] = last
                                s[i, j] = s[i, j - 1] - cv[it8] * t[
                                    int(s[i - 1, j - 1]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)]
                                ss[int((i - 1) / 2), j] = ss[int((i - 1) / 2), j - 1] + t[
                                    int(s[i - 1, j - 1]) - 1, int(s[i - 1, j]) - 1, int((i - 1) / 2)] + CD[
                                                              int((i - 1) / 2), j - 1]
                            else:
                                llast = s[i - 1, j + 1]
                                s[i - 1, j + 1] = last
                                s[i, j + 1] = s[i, j] - cv[it8] * t[
                                    int(s[i - 1, j]) - 1, int(s[i - 1, j + 1]) - 1, int((i - 1) / 2)]
                                ss[int((i - 1) / 2), j + 1] = ss[int((i - 1) / 2), j] + t[
                                    int(s[i - 1, j]) - 1, int(s[i - 1, j + 1]) - 1, int((i - 1) / 2)] + CD[
                                                                  int((i - 1) / 2), j]

                                if j + 1 == (it8 + 1) * (2 + patients + rs + 1) - 2:
                                    for j3 in range(j, 0, -1):
                                        if s[i - 1, j3] == 999:
                                            pre2 = j3
                                            break
                                    ad = 0
                                    for j4 in range(pre2 + 1, (it8 + 1) * (2 + patients + rs + 1) - 1):
                                        s[i - 1, pre2 + ad] = s[i - 1, j4]
                                        s[i, pre2 + ad] = s[i, j4]
                                        ss[int((i - 1) / 2), pre2 + ad] = ss[int((i - 1) / 2), j4]
                                        ad = ad + 1
                                    s[i - 1, j + 1] = llast
                                    s[i, j + 1] = s[i, j] - cv[it8] * t[
                                        int(s[i - 1, j]) - 1, int(s[i - 1, j + 1]) - 1, int((i - 1) / 2)]
                                    if s[i - 1, j] in rstation:
                                        ss[int((i - 1) / 2), j + 1] = ss[int((i - 1) / 2), j] + t[
                                            int(s[i - 1, j]) - 1, int(s[i - 1, j + 1]) - 1, int((i - 1) / 2)] + CD[
                                                                          int((i - 1) / 2), j]
                                    else:
                                        ss[int((i - 1) / 2), j + 1] = ss[int((i - 1) / 2), j] + t[
                                            int(s[i - 1, j]) - 1, int(s[i - 1, j + 1]) - 1, int((i - 1) / 2)] + DV[
                                                                          int(s[i - 1, j]) - 1]

                    if j == ((it8 + 1) * (2 + patients + rs + 1) - 2):
                        it8 = it8 + 1

    return (s, ss, h)

# REMOVE DUPLICATE SOLUTIONS
def remove_none(list):
    res = []
    for val in list:
        if val != None :
            res.append(val)
    return(res)

def remove_dup(pop, starttime, fitness):
    for i1 in range(len(fitness)):
        for i2 in range(i1):
            if fitness[i1] == fitness[i2]:
                fitness[i1] = None
                pop[i1] = np.array(None)
                starttime[i1] = np.array(None)
    #print("pop is:")
    #print(pop)
    fitness = [i for i in fitness if i]
    pop = [i for i in pop if i.any()]
    starttime = [i for i in starttime if i.any()]
    #print("pop22 is:")
    #print(pop)
    return (pop, starttime, fitness)

# FITNESS FUNCTIONS

def objectives0(s, ss, h, periods, patients, rs, pharmacies, drug,vehicles, t):
    z111 = 0
    z112 = 0
    z11 = 0
    z12 = 0
    z13 = 0
    z14 = 0
    z15 = 0
    z16 = 0
    z2 = 0
    z21 = 0
    z22 = 0
    BIG = 999999
    notnode = [999]
    for iii in range(pharmacies + patients + 1, pharmacies + patients + rs + 1):
        notnode.append(iii)
    # mc = np.zeros((periods, patients + rs + pharmacies + labs))
    #print(h)
    mc = np.zeros((periods, (vehicles * (2 + patients + rs)) + (vehicles - 1)))
    for i in range(mc.shape[0]):
        for j in range(mc.shape[1]):
            if h[i, j] != 0:
                mc[i, j] = 1
            else:
                mc[i, j] = 0
    #print(mc)
    ph = np.zeros((periods, vehicles))
    for i in range(ph.shape[0]):
        for j in range(ph.shape[1]):
            ph[i, j] = s[int(i * 2), int(j * ((2 + patients + rs) + 1))]
        #print(ph)
        #for j1 in range(ph.shape[1]):
        #    for j2 in range(j1):
        #        if ph[i, j1] == ph[i, j2]:
        #            ph[i, j1] = 0
        #print(ph)
        for j3 in range(ph.shape[1]):
            hint = 0
            for j4 in range(s.shape[1]):
                if (j3 * (2 + patients + rs + 1) + 1) <= j4 <= ((j3 + 1) * (2 + patients + rs + 1) - 3):
                    if s[int(i * 2), j4] not in notnode:
                        hint = hint + 1
            if hint == 0:
                ph[i, j3] = 0
        #print(ph)
    #for i1 in range(ph.shape[0]):
    #    for i2 in range(i1):
    #        for j in range(ph.shape[1]):
    #            if ph[i1, j] in (ph[:, :]).tolist()[i2]:
    #                ph[i1, j] = 0
    ph1 = ph.reshape(1,(periods*vehicles))
    ph1 = np.unique(ph1)
    print(ph1)
    #print(ph)
    for i in range(s.shape[0]):
        if i % 2 == 0:
            it9 = 0
            for j in range(s.shape[1]):
                if j != (it9 * (2 + patients + rs + 1)) and j != ((it9 + 1) * (2 + patients + rs + 1) - 1) and s[i, j] != 999:

                    #if j == it9 * (patients + rs + 2 + 1) + 1:
                        #pre4 = it9 * (patients + rs + 2 + 1)
                    #if (it9 * (2 + patients + rs + 1) + 1) <= j <= ((it9 + 1) * (2 + patients + rs + 1) - 2):
                    for j5 in range(j - 1, it9 * (patients + rs + 2 + 1) - 1, -1):
                        if s[i, j5] != 999:
                            pre4 = j5
                            break
                    z111 = z111 + t[int(s[i, pre4]) - 1, int(s[i, j]) - 1, int(i / 2)] * TC[it9]
                    z112 = z112 + + (s[i + 1, j] - h[int(i / 2), j]) * CC[it9] * mc[int(i / 2), j]
                    z11 = z111 + z112
                    #print("Distance:{}".format(j))
                    #print(z111)
                    #print(h[int(i / 2), j])
                    #print(s[i + 1, j])
                if j == ((it9 + 1) * (2 + patients + rs + 1) - 1):
                    it9 = it9 + 1



            #for jj in range(ph.shape[1]):
                #if ph[int(i / 2), jj] != 0:
                    #z12 = z12 + OCP[int(ph[int(i / 2), jj] - 1)]

    for jj in range(len(ph1)):
        if ph1[jj] != 0:
            z12 = z12 + OCP[int(ph1[jj] - 1)]
    D = np.zeros((pharmacies, drug, periods))
    it10 = 0

    for i in range(s.shape[0]):
        if i % 2 == 0:
            for d in range(drug):
                for j in range(s.shape[1]):
                    if j != it10 * ((2 + patients + rs) + 1):
                        if (it10 * (2 + patients + rs + 1) + 1) <= j <= ((it10 + 1) * (2 + patients + rs + 1) - 3):
                            if s[i, j] not in notnode:
                                # for d in range(drug):
                                D[int(s[i, int(it10 * ((2 + patients + rs) + 1))] - 1), d, int(i / 2)] = D[int(
                                    s[i, int(it10 * ((2 + patients + rs) + 1))] - 1), d, int(i / 2)] + A[int(
                                    s[i, j] - pharmacies - 1), d]

                if j == ((it10 + 1) * (2 + patients + rs + 1) - 3):
                    it10 = it10 + 1

    for i in range(D.shape[0]):
        for d in range(D.shape[1]):
            for p in range(D.shape[2]):
                z13 = z13 + max(((D[i, d, p] / CAP[i, d, p]) - 1), 0) * BIG

    it11 = 0
    for i in range(s.shape[0]):
        if i % 2 == 0:
            for j in range(s.shape[1]):
                if j != it11 * ((2 + patients + rs) + 1):
                    if (it11 * (2 + patients + rs + 1) + 1) <= j <= ((it11 + 1) * (2 + patients + rs + 1) - 3):
                        if s[i, j] not in notnode:
                            z14 = z14 + max(((q[int(s[i, j] - pharmacies - 1)] / qp[it11]) - 1), 0) * BIG
                            z15 = z15 + max(((ss[int(i / 2), j] / L[int(i / 2), int(s[i, j] - pharmacies - 1)]) - 1),
                                            0) * BIG
                            z16 = z16 + max(((E[int(i / 2), int(s[i, j] - pharmacies - 1)] / ss[int(i / 2), j]) - 1),
                                            0) * BIG
                if j == ((it11 + 1) * (2 + patients + rs + 1) - 3):
                    it11 = it11 + 1

    z1 = z11 + z12 + z13 + z14 + z15 + z16
    # z1 = z11 + z12
    z1 = round(z1, 4)
    #print("Z is :")
    #print(z111)
    #print(z112)
    #print(z11)
    #print(z12)
    #print(z13)
    #print(z14)
    #print(z15)
    #print(z16)
    #print(z1)

    for jj in range(len(ph1)):
        if ph1[jj] != 0:
            #z2 = z2 + we*((firstz2msax - (JO[int(ph1[jj] - 1)] * UR[int(ph1[jj] - 1)])) / (firstz2msax - firstz2min)) + wei*((
            #            secondz2max - (ev[int(ph1[jj] - 1)] * (1 - rd[int(ph1[jj] - 1)]))) / (secondz2max - secondz2min))
            z21 = z21 + (JO[int(ph1[jj] - 1)] * UR[int(ph1[jj] - 1)])
            z22 = z22 + (ev[int(ph1[jj] - 1)] * (1 - rd[int(ph1[jj] - 1)]))
            z21 = round(z21, 4)
            z22 = round(z22, 4)
    z2 = we*((firstz2msax - z21) / (firstz2msax - firstz2min)) + wei*((secondz2max - z22) / (secondz2max - secondz2min))
    z2 = round(z2, 4)
    return (z1, z2, z13,z14,z15,z16)
def objectives(s, ss, h, periods, patients, rs, pharmacies, drug,vehicles, t):
    z111 = 0
    z112 = 0
    z11 = 0
    z12 = 0
    z13 = 0
    z14 = 0
    z15 = 0
    z16 = 0
    z2 = 0
    z21 = 0
    z22 = 0
    BIG = 999999
    notnode = [999]
    for iii in range(pharmacies + patients + 1, pharmacies + patients + rs + 1):
        notnode.append(iii)
    # mc = np.zeros((periods, patients + rs + pharmacies + labs))
    #print(h)
    mc = np.zeros((periods, (vehicles * (2 + patients + rs)) + (vehicles - 1)))
    for i in range(mc.shape[0]):
        for j in range(mc.shape[1]):
            if h[i, j] != 0:
                mc[i, j] = 1
            else:
                mc[i, j] = 0
    #print(mc)
    ph = np.zeros((periods, vehicles))
    for i in range(ph.shape[0]):
        for j in range(ph.shape[1]):
            ph[i, j] = s[int(i * 2), int(j * ((2 + patients + rs) + 1))]
        #print(ph)
        #for j1 in range(ph.shape[1]):
        #    for j2 in range(j1):
        #        if ph[i, j1] == ph[i, j2]:
        #            ph[i, j1] = 0
        #print(ph)
        for j3 in range(ph.shape[1]):
            hint = 0
            for j4 in range(s.shape[1]):
                if (j3 * (2 + patients + rs + 1) + 1) <= j4 <= ((j3 + 1) * (2 + patients + rs + 1) - 3):
                    if s[int(i * 2), j4] not in notnode:
                        hint = hint + 1
            if hint == 0:
                ph[i, j3] = 0
        #print(ph)
    #for i1 in range(ph.shape[0]):
    #    for i2 in range(i1):
    #        for j in range(ph.shape[1]):
    #            if ph[i1, j] in (ph[:, :]).tolist()[i2]:
    #                ph[i1, j] = 0
    ph1 = ph.reshape(1,(periods*vehicles))
    ph1 = np.unique(ph1)
    #print(ph1)
    #print(ph)
    for i in range(s.shape[0]):
        if i % 2 == 0:
            it9 = 0
            for j in range(s.shape[1]):
                if j != (it9 * (2 + patients + rs + 1)) and j != ((it9 + 1) * (2 + patients + rs + 1) - 1) and s[i, j] != 999:

                    #if j == it9 * (patients + rs + 2 + 1) + 1:
                        #pre4 = it9 * (patients + rs + 2 + 1)
                    #if (it9 * (2 + patients + rs + 1) + 1) <= j <= ((it9 + 1) * (2 + patients + rs + 1) - 2):
                    for j5 in range(j - 1, it9 * (patients + rs + 2 + 1) - 1, -1):
                        if s[i, j5] != 999:
                            pre4 = j5
                            break
                    z111 = z111 + t[int(s[i, pre4]) - 1, int(s[i, j]) - 1, int(i / 2)] * TC[it9]
                    z112 = z112 + + (s[i + 1, j] - h[int(i / 2), j]) * CC[it9] * mc[int(i / 2), j]
                    z11 = z111 + z112
                    #print("Distance:{}".format(j))
                    #print(z111)
                    #print(h[int(i / 2), j])
                    #print(s[i + 1, j])
                if j == ((it9 + 1) * (2 + patients + rs + 1) - 1):
                    it9 = it9 + 1



            #for jj in range(ph.shape[1]):
                #if ph[int(i / 2), jj] != 0:
                    #z12 = z12 + OCP[int(ph[int(i / 2), jj] - 1)]

    for jj in range(len(ph1)):
        if ph1[jj] != 0:
            z12 = z12 + OCP[int(ph1[jj] - 1)]
    D = np.zeros((pharmacies, drug, periods))
    it10 = 0

    for i in range(s.shape[0]):
        if i % 2 == 0:
            for d in range(drug):
                for j in range(s.shape[1]):
                    if j != it10 * ((2 + patients + rs) + 1):
                        if (it10 * (2 + patients + rs + 1) + 1) <= j <= ((it10 + 1) * (2 + patients + rs + 1) - 3):
                            if s[i, j] not in notnode:
                                # for d in range(drug):
                                D[int(s[i, int(it10 * ((2 + patients + rs) + 1))] - 1), d, int(i / 2)] = D[int(
                                    s[i, int(it10 * ((2 + patients + rs) + 1))] - 1), d, int(i / 2)] + A[int(
                                    s[i, j] - pharmacies - 1), d]

                if j == ((it10 + 1) * (2 + patients + rs + 1) - 3):
                    it10 = it10 + 1

    for i in range(D.shape[0]):
        for d in range(D.shape[1]):
            for p in range(D.shape[2]):
                z13 = z13 + max(((D[i, d, p] / CAP[i, d, p]) - 1), 0) * BIG

    it11 = 0
    for i in range(s.shape[0]):
        if i % 2 == 0:
            for j in range(s.shape[1]):
                if j != it11 * ((2 + patients + rs) + 1):
                    if (it11 * (2 + patients + rs + 1) + 1) <= j <= ((it11 + 1) * (2 + patients + rs + 1) - 3):
                        if s[i, j] not in notnode:
                            z14 = z14 + max(((q[int(s[i, j] - pharmacies - 1)] / qp[it11]) - 1), 0) * BIG
                            z15 = z15 + max(((ss[int(i / 2), j] / L[int(i / 2), int(s[i, j] - pharmacies - 1)]) - 1),
                                            0) * BIG
                            z16 = z16 + max(((E[int(i / 2), int(s[i, j] - pharmacies - 1)] / ss[int(i / 2), j]) - 1),
                                            0) * BIG
                if j == ((it11 + 1) * (2 + patients + rs + 1) - 3):
                    it11 = it11 + 1

    z1 = z11 + z12 + z13 + z14 + z15 + z16
    # z1 = z11 + z12
    z1 = round(z1, 4)
    #print("Z is :")
    #print(z111)
    #print(z112)
    #print(z11)
    #print(z12)
    #print(z13)
    #print(z14)
    #print(z15)
    #print(z16)
    #print(z1)

    for jj in range(len(ph1)):
        if ph1[jj] != 0:
            # z2 = z2 + we*((firstz2msax - (JO[int(ph1[jj] - 1)] * UR[int(ph1[jj] - 1)])) / (firstz2msax - firstz2min)) + wei*((
            #            secondz2max - (ev[int(ph1[jj] - 1)] * (1 - rd[int(ph1[jj] - 1)]))) / (secondz2max - secondz2min))
            z21 = z21 + (JO[int(ph1[jj] - 1)] * UR[int(ph1[jj] - 1)])
            z22 = z22 + (ev[int(ph1[jj] - 1)] * (1 - rd[int(ph1[jj] - 1)]))
            z21 = round(z21, 4)
            z22 = round(z22, 4)
    z2 = we * ((firstz2msax - z21) / (firstz2msax - firstz2min)) + wei * (
                (secondz2max - z22) / (secondz2max - secondz2min))
    z2 = round(z2, 4)
    return (z1, z2)

# NON-DOMINATED SORTING
def fronts(popsize, fitness):
    F = {1: []}
    Sp = {k: [] for k in range(popsize)}
    np = {k: 0 for k in range(popsize)}
    for p in range(popsize):
        for q in range(popsize):
            if p != q:
                if (fitness[p][0] < fitness[q][0] and fitness[p][1] < fitness[q][1]) or (
                        fitness[p][0] <= fitness[q][0] and fitness[p][1] < fitness[q][1]) or (
                        fitness[p][0] < fitness[q][0] and fitness[p][1] <= fitness[q][1]):
                    Sp[p].append(q)
                elif (fitness[p][0] > fitness[q][0] and fitness[p][1] > fitness[q][1]) or (
                        fitness[p][0] >= fitness[q][0] and fitness[p][1] > fitness[q][1]) or (
                        fitness[p][0] > fitness[q][0] and fitness[p][1] >= fitness[q][1]):
                    np[p] = np[p] + 1

        if np[p] == 0:
            F[1].append(p)

    i = 1
    Q = [1]
    while Q != []:
        Q = []
        for p in F[i]:
            for q in Sp[p]:
                np[q] = np[q] - 1
                if np[q] == 0:
                    Q.append(q)

        i = i + 1
        F[i] = Q
    return (F)

# CROWDING DISTANCE
def crowding_distance0(popsize, fitness):
    crowd = {k: 0 for k in range(popsize)}
    f1 = {k: 0 for k in range(popsize)}
    f2 = {k: 0 for k in range(popsize)}
    for i in range(len(fitness)):
        f1[i] = fitness[i][0]
    for i in range(len(fitness)):
        f2[i] = fitness[i][1]
    f1sorted = {k: v for k, v in sorted(f1.items(), key=lambda item: item[1])}
    f2sorted = {k: v for k, v in sorted(f2.items(), key=lambda item: item[1])}

    first1 = min(f1sorted, key=lambda k: f1sorted[k])
    last1 = max(reversed(range(len(f1sorted))), key=f1sorted.__getitem__)

    first2 = min(f2sorted, key=lambda k: f2sorted[k])
    # last2 = max(f2sorted, key=lambda k: f2sorted[k])
    # last2 = max(enumerate(f2sorted), key=lambda x: (x[1], x[0]))[0]
    last2 = max(reversed(range(len(f2sorted))), key=f2sorted.__getitem__)

    lis1 = list(f1sorted)
    lis2 = list(f2sorted)

    # print(f1sorted)
    # print(first1)
    # print(last1)
    # print(f2sorted)
    # print(first2)
    # print(last2)
    # mm= lis1.index(0)
    # print(mm)
    # kk = lis1[mm-1]
    # print(f1sorted[kk])

    for p in range(popsize):
        if p != first1 and p != last1:
            i = lis1.index(p)
            crowd[p] = (f1sorted[lis1[i + 1]] - f1sorted[lis1[i - 1]]) / (f1sorted[last1] - f1sorted[first1])
            crowd[p] = round(crowd[p], 4)

    for p in range(popsize):
        if p != first2 and p != last2:
            i = lis2.index(p)
            crowd[p] = crowd[p] + (f2sorted[lis2[i + 1]] - f2sorted[lis2[i - 1]]) / (f2sorted[last2] - f2sorted[first2])
            crowd[p] = round(crowd[p], 4)

    return (crowd)
def crowding_distance(popsize, fitness, F):
    crowd = {k: 0 for k in range(popsize)}
    f1 = {k: 0 for k in range(popsize)}
    f2 = {k: 0 for k in range(popsize)}
    for i in range(len(fitness)):
        f1[i] = fitness[i][0]
    for i in range(len(fitness)):
        f2[i] = fitness[i][1]
    #print("CROWDING DISTANCE")
    #print(F)
    #print(f1)
    #print(f2)
    for i in list(F.keys()):
        if i != list(F)[-1]:
            f11 = {k: f1[k] for k in list(F.values())[i - 1]}
            f22 = {k: f2[k] for k in list(F.values())[i - 1]}
            #print(f11)
            #print(f22)
            f1sorted = {k: v for k, v in sorted(f11.items(), key=lambda item: item[1])}
            f2sorted = {k: v for k, v in sorted(f22.items(), key=lambda item: item[1])}
            #print(f1sorted)
            #print(f2sorted)
            first1 = min(f1sorted, key=lambda k: f1sorted[k])
            #print(first1)
            #last1 = max(reversed(f1sorted),key=lambda k: f1sorted[k])
            last1 = max(f1sorted, key=lambda k: f1sorted[k])
            #print(last1)
            first2 = min(f2sorted, key=lambda k: f2sorted[k])
            #last2 = max(reversed(f2sorted), key=lambda k: f2sorted[k])
            last2 = max(f2sorted, key=lambda k: f2sorted[k])
            lis1 = list(f1sorted)
            lis2 = list(f2sorted)
            #print(first2)
            #print(last2)
            for j in range(len(F[i])):
                if F[i][j] != first1 and F[i][j] != last1:
                    ii = lis1.index(F[i][j])
                    crowd[F[i][j]] = (f1sorted[lis1[ii + 1]] - f1sorted[lis1[ii - 1]]) / (f1sorted[last1] - f1sorted[first1])
                    crowd[F[i][j]] = round(crowd[F[i][j]], 4)

            for j in range(len(F[i])):
                if F[i][j] != first2 and F[i][j] != last2:
                    ii = lis2.index(F[i][j])
                    crowd[F[i][j]] = crowd[F[i][j]] + (f2sorted[lis2[ii + 1]] - f2sorted[lis2[ii - 1]]) / (f2sorted[last2] - f2sorted[first2])
                    crowd[F[i][j]] = round(crowd[F[i][j]], 4)
                else:
                    crowd[F[i][j]]= 99999
            #print("CD")
            #print(crowd)
    return (crowd)

# PARENT SELECTION & CROSS OVER & MUTATION

# CROSS OVER
def cross_over(periods, patients, rs, vehicles, pop, cr):
    popoffcr = []
    starttimeoffcr = []
    fitnessoffcr = []
    # cr = 0.75
    # par1 = random.choice(pop)
    # par2 = random.choice(pop)
    #print("popsizeee:")
    #print(len(pop))
    for i in range(int(math.ceil(math.ceil(cr * len(pop)) / 2))):
        r = (random.randint(0, len(pop) - 1))
        rr = r
        while r == rr:
            rr = (random.randint(0, len(pop) - 1))
        #print(r)
        #print(rr)
        par1 = pop[r]
        par2 = pop[rr]
        #print("Parent 1 is: \n")
        #print(par1)
        #print("Parent 2 is: \n")
        #print(par2)
        mask = np.zeros((periods, (vehicles * (2 + patients + rs)) + (vehicles - 1)))
        for j in range(mask.shape[1]):
            mask[0, j] = np.random.randint(2)
        for i in range(mask.shape[0]):
            mask[i, :] = mask[0, :]
        #print("Mask matrix is: \n")
        #print(mask)
        off1 = np.zeros((periods * 2, (vehicles * (2 + patients + rs)) + (vehicles - 1)))
        off2 = np.zeros((periods * 2, (vehicles * (2 + patients + rs)) + (vehicles - 1)))
        for i in range(off1.shape[0]):
            if i % 2 == 0:
                for j in range(off1.shape[1]):
                    if mask[int(i / 2), j] == 1:
                        off1[i, j] = par1[i, j]
                    else:
                        off1[i, j] = par2[i, j]
        for i in range(off2.shape[0]):
            if i % 2 == 0:
                for j in range(off2.shape[1]):
                    if mask[int(i / 2), j] == 1:
                        off2[i, j] = par2[i, j]
                    else:
                        off2[i, j] = par1[i, j]

        #print("Offspring1 is: \n")
        #print(off1)
        #print("Offspring2 is: \n")
        #print(off2)
        s1, ss1, h1 = adjust_solution(s=off1,periods=periods, patients=patients, rs=rs, pharmacies=pharmacies, labs=labs,vehicles=vehicles, cv=cv, t=t, sch=sch,rstation=rstation)
        popoffcr.append(s1)
        starttimeoffcr.append(ss1)
        f1 = objectives(s=s1, ss=ss1, h=h1, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,
                        vehicles=vehicles,drug=drug, t=t)
        fitnessoffcr.append(f1)
        s2, ss2, h2 = adjust_solution(s=off2,periods=periods, patients=patients, rs=rs, pharmacies=pharmacies, labs=labs,vehicles=vehicles, cv=cv, t=t, sch=sch,rstation=rstation)
        popoffcr.append(s2)
        starttimeoffcr.append(ss2)
        f2 = objectives(s=s2, ss=ss2, h=h2, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,
                        vehicles=vehicles,drug=drug, t=t)
        fitnessoffcr.append(f2)
        #print("Adjusted Offspring1 is: \n")
        #print(s1)
        #print("Adjusted Offspring2 is: \n")
        #print(s2)

    if len(popoffcr) != math.ceil(cr * len(pop)):
        popoffcr.pop()
        starttimeoffcr.pop()
        fitnessoffcr.pop()
    return (popoffcr, starttimeoffcr, fitnessoffcr)

# MUTATION
def mutation(patients, rs, pop, mr, numcol_s):
    popoffmu = []
    starttimeoffmu = []
    fitnessoffmu = []
    # mr = 0.25
    it12 = 0
    validlist = []
    for i in range(numcol_s):
        if i != (it12 * ((2 + patients + rs) + 1)) and i != ((it12 + 1) * ((2 + patients + rs) + 1)) - 2 and i != (
                (it12 + 1) * ((2 + patients + rs) + 1)) - 1:
            validlist.append(i)
        if i == ((it12 + 1) * ((2 + patients + rs) + 1)) - 1:
            it12 = it12 + 1

    for i in range(math.floor(mr * len(pop))):
        r = random.randint(0, len(pop) - 1)
        off3 = pop[r]
        #print("Offspring3 is from this parent: \n")
        #print(off3)
        i1 = random.choice(validlist)
        i2 = i1
        while i1 == i2:
            i2 = random.choice(validlist)
        # print(i1)
        # print(i2)
        for ii in range(off3.shape[0]):
            if ii % 2 == 0:
                off3[ii, i1], off3[ii, i2] = off3[ii, i2], off3[ii, i1]
        #print("Offspring3 is: \n")
        #print(off3)
        s3, ss3, h3 = adjust_solution(s=off3,periods=periods, patients=patients, rs=rs, pharmacies=pharmacies, labs=labs,vehicles=vehicles, cv=cv, t=t, sch=sch,rstation=rstation)
        #print("Adjusted Offspring3 is: \n")
        #print(s3)
        popoffmu.append(s3)
        starttimeoffmu.append(ss3)
        f3 = objectives(s=s3, ss=ss3, h=h3, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,
                        vehicles=vehicles,drug=drug, t=t)
        fitnessoffmu.append(f3)


    return (popoffmu, starttimeoffmu, fitnessoffmu)

# LOCAL SEARCHES
def local2(s,patients, rs):
    #popoffl2 = []
    #starttimeoffl2 = []
    #fitnessoffl2 = []
    phar = s[0, 0]
    for i in range(s.shape[0]):
        if i % 2 == 0:
            it1 = 0
            for j in range(s.shape[1]):
                if j == (it1 * ((2 + patients + rs) + 1)):
                    s[i, j] = phar
                    it1 = it1 + 1

    s5, ss5, h5 = adjust_solution(s=s, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies, labs=labs,vehicles=vehicles, cv=cv, t=t, sch=sch, rstation=rstation)
    #popoffl2.append(s5)
    #starttimeoffl2.append(ss5)
    f5 = objectives(s=s5, ss=ss5, h=h5, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,vehicles=vehicles, drug=drug, t=t)
    #fitnessoffl2.append(f5)
    return (s5, ss5, f5)

def local4(s,patients, rs):
    #popoffl4 = []
    #starttimeoffl4 = []
    #fitnessoffl4 = []
    for i in range(s.shape[0]):
        if i % 2 == 0:
            it1 = 0
            for j in range(s.shape[1]):
                if j == it1 * (2 + patients + rs + 1):
                    ph1 = s[i, j]
                    #print("ph1:")
                    #print(ph1)
                    ph2 = ph1
                    while ph2 == ph1:
                        ph2 = (random.randint(1, pharmacies))
                    #print("ph2:")
                    #print(ph2)
                    s[i, j] = ph2
                    it1 = it1 + 1

    s7, ss7, h7 = adjust_solution(s=s, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies, labs=labs,vehicles=vehicles, cv=cv, t=t, sch=sch, rstation=rstation)
    # popoffl4.append(s7)
    # starttimeoffl4.append(ss7)
    f7 = objectives(s=s7, ss=ss7, h=h7, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,vehicles=vehicles, drug=drug, t=t)
    # fitnessoffl4.append(f7)

    return (s7, ss7, f7)

# DEVELOPING THE NSGA II

def generate_feasible(periods, patients, rs,pharmacies, labs,vehicles, cv, t, sch, rstation,pop,starttime,fitness):
    while True:
        s0, ss0, h0 = generate_solution(periods=periods, patients=patients, rs=rs, pharmacies=pharmacies, labs=labs,
                                        vehicles=vehicles, cv=cv, t=t, sch=sch, rstation=rstation)
        rand = random.uniform(0, 1)
        if rand < 0.6:
            s1, ss1, f1 = local4(s=s0, patients=patients, rs=rs)
            s0, ss0, h0 = adjust_solution(s=s1, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,
                                          labs=labs,
                                          vehicles=vehicles, cv=cv, t=t, sch=sch, rstation=rstation)
        if rand < 0.4:
            s1, ss1, f1 = local2(s=s0, patients=patients, rs=rs)
            s0, ss0, h0 = adjust_solution(s=s1, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,
                                          labs=labs,
                                          vehicles=vehicles, cv=cv, t=t, sch=sch, rstation=rstation)
        f0 = objectives0(s=s0, ss=ss0, h=h0, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,vehicles=vehicles, drug=drug, t=t)

        if (f0[2] == 0) and (f0[3] == 0) and (f0[4] == 0) and (f0[5] == 0):
            break
    print(f0)
    pop.append(s0)
    starttime.append(ss0)
    f = objectives(s=s0, ss=ss0, h=h0, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,vehicles=vehicles, drug=drug, t=t)
    fitness.append(f)
    pop, starttime, fitness = remove_dup(pop=pop, starttime=starttime, fitness=fitness)
    return (pop, starttime, fitness)

def NSGAII(periods, patients, rs, pharmacies, labs, vehicles, cv, t, sch, popsize, cr, mr, generations):
    pop = []
    starttime = []
    fitness = []
    # popsize = 4

    i = 1
    while i <= popsize:
        while True:
            s0, ss0, h0 = generate_solution(periods=periods, patients=patients, rs=rs, pharmacies=pharmacies, labs=labs,
                                            vehicles=vehicles, cv=cv, t=t, sch=sch, rstation=rstation)
            rand = random.uniform(0, 1)
            if rand < 0.3:
                s1, ss1, f1 = local2(s=s0, patients=patients, rs=rs)
                s0, ss0, h0 = adjust_solution(s=s1, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,
                                              labs=labs,
                                              vehicles=vehicles, cv=cv, t=t, sch=sch, rstation=rstation)
            f0 = objectives0(s=s0, ss=ss0, h=h0, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,
                             vehicles=vehicles, drug=drug, t=t)
            # print(s0)
            # print(ss0)
            print(f0)
            if (f0[2] == 0) and (f0[3] == 0) and (f0[4] == 0) and (f0[5] == 0):
                break

        pop.append(s0)
        starttime.append(ss0)
        f = objectives(s=s0, ss=ss0, h=h0, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,
                       vehicles=vehicles, drug=drug, t=t)
        # ff = objectives0(s=s0, ss=ss0, h=h0, periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,
        #               vehicles=vehicles, drug=drug, t=t)
        # print(ff)
        fitness.append(f)
        # print(fitness)
        pop, starttime, fitness = remove_dup(pop=pop, starttime=starttime, fitness=fitness)
        i = i + 1
    print("Initial population is :\n")
    print(pop)
    #generations = 1
    bests = {k: 0 for k in range(generations)}

    for g in range(generations):
        nextpop = []
        nextstarttime = []
        nextfitness = []
        # print(pop)

        popoffcr, starttimeoffcr, fitnessoffcr = cross_over(periods=periods, patients=patients, rs=rs,
                                                            vehicles=vehicles, pop=pop, cr=cr)
        popoffmu, starttimeoffmu, fitnessoffmu = mutation(patients=patients, rs=rs, pop=pop, mr=mr,
                                                          numcol_s=vehicles * (2 + patients + rs) + vehicles - 1)
        # popoffl, starttimeoffl, fitnessoffl = local(sch= sch, patients=patients, rs=rs, vehicles=vehicles, pop=pop, lr=lr)
        # popoffl3, starttimeoffl3, fitnessoffl3 = local3(patients= patients, rs=rs, vehicles=vehicles, pop=pop, l3r=l3r)
        # popoffl4, starttimeoffl4, fitnessoffl4 = local4(patients=patients, rs=rs, pop=pop, l4r=l4r)

        uppop = pop + popoffcr + popoffmu
        upstarttime = starttime + starttimeoffcr + starttimeoffmu
        upfitness = fitness + fitnessoffcr + fitnessoffmu

        uppop, upstarttime, upfitness = remove_dup(pop=uppop, starttime=upstarttime, fitness=upfitness)
        print("Population after {} generation is:".format(g))
        # print(uppop)
        # print("Starttime after {} generation is:".format(g))
        # print(upstarttime)
        # print("Fitness after {} generation is:".format(g))
        # print(upfitness)
        F = fronts(popsize=len(uppop), fitness=upfitness)
        C = crowding_distance(popsize=len(uppop), fitness=upfitness, F=F)
        print("Fronts after {} generation is:".format(g))
        # print(F)
        # print("Crowding distance after {} generation is:".format(g))
        # print(C)

        for k in F.keys():
            F[k] = sorted(F[k], key=lambda i: C[i], reverse=True)

        for s in F.keys():
            for r in range(len(F[s])):
                if len(nextpop) < len(pop):
                    nextpop.append(uppop[F[s][r]])
                    nextstarttime.append(upstarttime[F[s][r]])
                    nextfitness.append(upfitness[F[s][r]])

        pop = nextpop
        starttime = nextstarttime
        fitness = nextfitness
        bests[g] = fitness[0][0]
        # print("Initial population of {} generation is:".format(g + 1))
        # print(pop)
        # print("Starttime of initial population of {} generation is:".format(g+1))
        # print(starttime)
        # print("Fitness of initial population of {} generation is:".format(g+1))
        # print(fitness)
    return (upfitness, F, C, pop, starttime, fitness)


from datetime import datetime

start = datetime.now()
upfitness, F, C, pop, starttime, fitness = NSGAII(periods=periods, patients=patients, rs=rs, pharmacies=pharmacies,
                                                  labs=labs, vehicles=vehicles, cv=cv, t=t, sch=sch, popsize=70,
                                                  cr=0.65, mr=0.15, generations=20)

print(datetime.now() - start)
# print(pop)
print(F)
print(C)
print(upfitness)
print(fitness)
flist = []
for s in F[1]:
    flist.append(upfitness[s])
print(flist)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(fitness)):
    x = fitness[i][0]
    y = fitness[i][1]
    ax.scatter(x, y)
plt.show()