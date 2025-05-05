#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math as math
from sympy import Matrix, linsolve, symbols
from sympy import symbols, nonlinsolve
from sympy.polys.polytools import is_zero_dimensional
from numpy.linalg import matrix_rank
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols, Eq, solve
import scipy as sp
import sympy as smp
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time
from scipy.optimize import fsolve
from tkinter import Tk, ttk, Canvas
from tkinter import Scrollbar
from tksheet import Sheet
import sys
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QPushButton
from PyQt5.QtGui import QColor
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.stats import norm
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import skew
import openpyxl
from openpyxl.styles import PatternFill
from itertools import product
from flask import Flask, send_file
import threading

# Ρύθμιση ακρίβειας δεκαδικών ψηφίων
getcontext().prec = 15

filename = "latest_table_paper2.xlsx"


# Ορίζουμε τα εύρη τιμών
r_values = np.linspace(0.01, 0.04, 31)
rho_v = np.linspace(-1, 1, 11)
d_v = np.linspace(0.2,0.01, 30)

m = np.log(0.1)
v = 1/np.sqrt(2)
k0 = 4
k1 = 2
c = 1
loko = 2

def f(y):
    return 0.05 + (np.tanh(loko * (y - m)) + 1) * (0.3 - 0.05) / 2

# Φτιάχνουμε τη λίστα
combinations = []

for rho in rho_v:
    for r in r_values:
        mi_values = np.arange(0.005, r - 0.001 + 1e-8, 0.001)
        for mi in mi_values:
            for d in d_v:
                # Στρογγυλοποιούμε ΜΟΝΟ όταν φτιάχνουμε το tuple
                combinations.append((
                    np.round(d, 5),
                    np.round(r, 5),
                    np.round(mi, 5),
                    np.round(rho, 5)
                ))

print(f"Συνολικά combinations: {len(combinations)}")


N=250
M=15

ppmax=2.5
ppmin=0


Pmax=ppmax-ppmin

yymax=m+3*v*v
yymin=m-3*v*v
Ymax=yymax-yymin
yrange=6*v*v



dp=Pmax/N
dy=Ymax/M

iterate=0

overall_start_time = time.time()

def run_all_combinations():
    for d,r , mi, rho in combinations:
        print(iterate+1,d,r , mi, rho)

        def f_bar_squared_limited(fok, mok, nu, lower_bound=-np.inf, upper_bound=np.inf):
            """
            Calculate the value of f_bar squared with finite integration bounds.

            Parameters:
            f (function): The volatility function f(y).
            m (float): Mean level of the Gaussian distribution.
            nu (float): Standard deviation related to the mean reversion speed.
            lower_bound (float): Lower bound for integration.
            upper_bound (float): Upper bound for integration.

            Returns:
            float: The approximated value of f_bar squared.
            """
            def integrand(y):
                return fok(y)**2 * (1 / (np.sqrt(2 * np.pi) * nu)) * np.exp(-((mok - y)**2) / (2 * nu**2))

            result, _ = quad(integrand, lower_bound, upper_bound)
            return result

        # Example usage
        f_bar_squared_approx = f_bar_squared_limited(f, m, v)
        fbar2 = f_bar_squared_approx
        ss2 = f_bar_squared_approx

        # Συνάρτηση ολοκληρώματος
        def omega(y,Cc=0):
            return -(0.175 * y + 0.125 * (1/loko) * np.log(np.cosh(loko * (y - m))) + Cc)

        # Περιορισμός της περιοχής ολοκλήρωσης
        lower_bound = -np.inf
        upper_bound = np.inf

        # Αριθμητική ολοκλήρωση
        def integrand(y):
            omega_val = omega(y)
            if np.isnan(omega_val) or np.isinf(omega_val):
                return 0  # Αποφυγή NaN ή Inf
            return omega_val * (ss2 - f(y)**2) * (1 / (np.sqrt(np.pi) * v)) * np.exp(-((m - y)**2) / (2 * v**2))

        INTE, _ = quad(integrand, lower_bound, upper_bound)

        G=(1/(2*v*v))*INTE* (1/np.sqrt(2))

        # Υπολογισμός παραμέτρων a, b
        a = (-mi + (ss2 / 2) + np.sqrt(2 * ss2 * r + (mi - ss2 / 2)**2)) / ss2
        b = (-mi + (ss2 / 2) - np.sqrt(2 * ss2 * r + (mi - ss2 / 2)**2)) / ss2

        # Υπολογισμός g
        g = (np.sqrt(2) * v * rho) / (ss2 * (a - b))


        # Ορισμός εξισώσεων
        A, B, P00, P10 = smp.symbols("A B P00 P10", real=True)

        eq1 = smp.Eq(A * P10**a - k1, B * P10**b + P10 / (r - mi) - c / r)
        eq2 = smp.Eq(A * P00**a + k0, B * P00**b + P00 / (r - mi) - c / r)
        eq3 = smp.Eq(A * a * P00**(a - 1), B * b * P00**(b - 1) + 1 / (r - mi))
        eq4 = smp.Eq(A * a * P10**(a - 1), B * b * P10**(b - 1) + 1 / (r - mi))

        # Επίλυση συστήματος εξισώσεων
        def func(x):
            return [
                x[0] * x[3]**a - k1 - x[1] * x[3]**b - x[3] / (r - mi) + c / r,
                x[0] * x[2]**a + k0 - x[1] * x[2]**b - x[2] / (r - mi) + c / r,
                x[0] * a * x[2]**(a - 1) - x[1] * b * x[2]**(b - 1) - 1 / (r - mi),
                x[0] * a * x[3]**(a - 1) - x[1] * b * x[3]**(b - 1) - 1 / (r - mi),
            ]

        root = fsolve(func, [90, 15, 4, 0.4])
        A, B, P00, P10 = root

        # Υπολογισμός σταθερών D0, D1, E0, E1, C0, C1
        D0 = A * a * a * (a - 1)
        D1 = B * b * b * (b - 1)
        E0 = D0 * P00**a + D1 * P00**b
        E1 = D0 * P10**a + D1 * P10**b

        C0 = (E0 * P10**b * np.log(P00) - E1 * P00**b * np.log(P10)) / (P00**b * P10**a - P00**a * P10**b)
        C1 = (E0 * P10**a * np.log(P00) - E1 * P00**a * np.log(P10)) / (P00**b * P10**a - P00**a * P10**b)

        # Ορισμός συναρτήσεων V00, V10, V01, V11
        def V00(pp):
            return A * pp**a

        def V10(pp):
            pp = float(pp)
            return B * pp**b + pp / (r - mi) - c / r

        def V01(pp):
            pp = float(pp)
            return g * G * pp**a * (D0 * np.log(pp) + C0)

        def V11(pp):
            pp = float(pp)
            return -g * G * pp**b * (D1 * np.log(pp) + C1)

        # Υπολογισμός P01 και P11
        P01 = a * b * g * G * (
            (D0 * P00**(a + 1) + D1 * P00**(b + 1)) / (a * D1 * P00**b - b * D0 * P00**a) +
            (P00**(a + b + 1) * (D0 * P10**a + D1 * P10**b) * (a - b) * np.log(P10 / P00)) /
            ((P00**a * P10**b - P00**b * P10**a) * (a * D1 * P00**b - b * D0 * P00**a))
        )

        P11 = a * b * g * G * (
            (D0 * P10**(a + 1) + D1 * P10**(b + 1)) / (a * D1 * P10**b - b * D0 * P10**a) +
            (P10**(a + b + 1) * (D0 * P00**a + D1 * P00**b) * (a - b) * np.log(P10 / P00)) /
            ((P00**a * P10**b - P00**b * P10**a) * (a * D1 * P10**b - b * D0 * P10**a))
        )

        # Υπολογισμός P0best και P1best
        P0best = P00 + d * P01
        P1best = P10 + d * P11

        # Υπολογισμός τελικών τιμών Vf0 και Vf1
        def Vf0(pp):
            return V00(pp) + d * V01(pp)

        def Vf1(pp):
            return V10(pp) + d * V11(pp)



        ###### ########### ########### ######### ######### ###### ########### ########### ######### #########
        M_l=M
        M_h=M

        Pmax=ppmax-ppmin

        yymax=m+3*v*v
        yymin=m-3*v*v
        Ymax=yymax-yymin
        yrange=6*v*v

        ###### ########### ########### ######### ######### ###### ########### ########### ######### #########

        M_l,M_h=M,M
        yymax_l,yymax_h=yymax,yymax
        yymin_l,yymin_h=yymin,yymin
        Ymax_l,Ymax_h=yymax-yymin,yymax-yymin

        dp=Pmax/N
        dy=Ymax/M

        letsl=yymin+yrange/6
        letsh=yymax-yrange/6

        print('y= (',yymin,',',yymax,')')


        def Af(p, y, dp_g, dy_g):
            return -(2 * (d ** -2) * v ** 2) / (dy_g ** 2) - ((f(y) ** 2) * (p ** 2)) / (dp_g ** 2) - r

        def Bf(p, y, dp_g, dy_g):
            return ((f(y) ** 2) * (p ** 2)) / (2 * dp_g ** 2) - (mi * p) / (2 * dp_g)
    
        def Cf(p, y, dp_g, dy_g):
            return -(d ** -2) * (m - y) / (2 * dy_g) + ((d ** -2) * v ** 2) / (dy_g ** 2)

        def Df(p, y, dp_g, dy_g):
            return ((f(y) ** 2) * (p ** 2)) / (2 * dp_g ** 2) + (mi * p) / (2 * dp_g)

        def Ef(p, y, dp_g, dy_g):
            return (d ** -2) * (m - y) / (2 * dy_g) + ((d ** -2) * v ** 2) / (dy_g ** 2)

        def Ff(p, y, dp_g, dy_g):
            return ((d ** -1) * np.sqrt(2) * v * rho * f(y) * p) / (4 * dp_g * dy_g)

        def Gf(p, y, dp_g, dy_g):
            return -((d ** -1) * np.sqrt(2) * v * rho * f(y) * p) / (4 * dp_g * dy_g)

        def Hf(p, y, dp_g, dy_g):
            return -((d ** -1) * np.sqrt(2) * v * rho * f(y) * p) / (4 * dp_g * dy_g)

        def If(p, y, dp_g, dy_g):
            return ((d ** -1) * np.sqrt(2) * v * rho * f(y) * p) / (4 * dp_g * dy_g)


        #V0panwd=np.ones(M+1)*((1/(r-mi))*Pmax-c/r-k0)
        V0panwd=np.ones(M+1)*(Vf1(Pmax)-k0)
        V0katwd=np.zeros(M+1)
        #V1panwd=np.ones(M+1)*((1/(r-mi))*Pmax-c/r)
        V1panwd=np.ones(M+1)*Vf1(Pmax)
        V1katwd=np.zeros(M+1)-k1

        def findi_n(p,y):
            return p*(M+1)+y

        def findpy_n(i):
           return (i//(M+1),i % (M+1))

        if 'x0_start' not in locals() or x0_start is None or iterate==0:
            x0_start=np.ones((N-1)*(M+1))
            x1_start=np.ones((N-1)*(M+1))
            for xyz in range((N-1)*(M+1)):
                ii,jj=findpy_n(xyz)
                if ii<=P1best:
                    x0_start[xyz]=Vf0((ii+1)*dp)
                    x1_start[xyz]=Vf0((ii+1)*dp)-k1
                elif ii>=P0best:
                    x0_start[xyz]=Vf1((ii+1)*dp)-k0
                    x1_start[xyz]=Vf1((ii+1)*dp)
                else:
                    x0_start[xyz]=Vf0((ii+1)*dp)
                    x1_start[xyz]=Vf1((ii+1)*dp)

        def Acalc(pppmax,pppmin,yyymax,yyymin,MM,NN,Vkatw,Vpanw,vari):
            dpp=(pppmax-pppmin)/NN
            dyy=(yyymax-yyymin)/MM
            TT=np.zeros(((MM+1)*(NN-1),(MM+1)*(NN-1)))
            bb=np.zeros((MM+1)*(NN-1))
            temp=0
            for pi in range(NN-1):
                for yi in range(MM+1):
                    pmet=pppmin+(pi+1)*dpp
                    ymet=yyymin+yi*dyy
                    if pi!=0 and yi!=0 and pi!=NN-2 and yi!=MM:
                        TT[temp,findi_n(pi,yi)]=Af(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi-1,yi)]=Bf(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi,yi-1)]=Cf(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi+1,yi)]=Df(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi,yi+1)]=Ef(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi+1,yi+1)]=Ff(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi+1,yi-1)]=Gf(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi-1,yi+1)]=Hf(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi-1,yi-1)]=If(pmet,ymet,dpp,dyy)
                        bb[temp]=0
                        bb[temp]=bb[temp]+vari*(-pmet+c)
                        temp=temp+1
                    elif pi==0 and yi!=0 and yi!=MM:
                        TT[temp,findi_n(pi,yi)]=Af(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi,yi-1)]=Cf(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi+1,yi)]=Df(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi,yi+1)]=Ef(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi+1,yi+1)]=Ff(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi+1,yi-1)]=Gf(pmet,ymet,dpp,dyy)
                        bb[temp]=0
                        bb[temp]=bb[temp]-Bf(pmet,ymet,dpp,dyy)*Vkatw[yi]
                        bb[temp]=bb[temp]-If(pmet,ymet,dpp,dyy)*Vkatw[yi-1]
                        bb[temp]=bb[temp]-Hf(pmet,ymet,dpp,dyy)*Vkatw[yi+1]
                        bb[temp]=bb[temp]+vari*(-pmet+c)
                        temp=temp+1
                    elif pi==NN-2 and yi!=0 and yi!=MM:
                        TT[temp,findi_n(pi,yi)]=Af(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi-1,yi)]=Bf(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi,yi-1)]=Cf(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi,yi+1)]=Ef(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi-1,yi-1)]=If(pmet,ymet,dpp,dyy)
                        TT[temp,findi_n(pi-1,yi+1)]=Hf(pmet,ymet,dpp,dyy)
                        bb[temp]=0
                        bb[temp]=bb[temp]+vari*(-pmet+c)
                        bb[temp]=bb[temp]-Df(pmet,ymet,dpp,dyy)*Vpanw[yi]
                        bb[temp]=bb[temp]-Ff(pmet,ymet,dpp,dyy)*Vpanw[yi+1]
                        bb[temp]=bb[temp]-Gf(pmet,ymet,dpp,dyy)*Vpanw[yi-1]
                        temp=temp+1
                    elif yi==0 :
                        TT[temp,findi_n(pi,yi)]=-1
                        TT[temp,findi_n(pi,yi+1)]=1
                        bb[temp]=0
                        temp=temp+1
                    elif yi==MM :
                        TT[temp,findi_n(pi,yi)]=1
                        TT[temp,findi_n(pi,yi-1)]=-1
                        bb[temp]=0
                        temp=temp+1
                    else:
                        print(pi,yi)
            return TT,bb

        def normalize_system(A, b):
            """
            Κανονικοποιεί το σύστημα γραμμικών εξισώσεων Ax = b.
            Διασφαλίζει ότι ο κάθε συντελεστής στη διαγώνιο του A είναι 1.

            :param A: Ο πίνακας συντελεστών (πλήρης ή αραιός).
            :param b: Το διάνυσμα αποτελεσμάτων.
            :return: Ο κανονικοποιημένος πίνακας A και το διάνυσμα b.
            """
            # Δημιουργία αντιγράφων για να μην επηρεαστούν τα αρχικά δεδομένα
            A_normalized = np.copy(A)
            b_normalized = np.copy(b)

            # Κανονικοποίηση ανά γραμμή
            for i in range(len(A)):
                diagonal_element = A[i, i]
                if diagonal_element != 0:
                    A_normalized[i] = A[i] / diagonal_element
                    b_normalized[i] = b[i] / diagonal_element
                else:
                    raise ValueError(f"Μηδενικό διαγώνιο στοιχείο στη γραμμή {i}.")
            return A_normalized, b_normalized

        T0,b0=Acalc(ppmax,ppmin,yymax,yymin,M,N,V0katwd,V0panwd,0)
        T1,b1=Acalc(ppmax,ppmin,yymax,yymin,M,N,V1katwd,V1panwd,1)

        T0_n,b0_n=normalize_system(T0,b0)
        T1_n,b1_n=normalize_system(T1,b1)


        #BEST PROGRAM 4

        def solve_two_same_size_systems(A0_prob, b0_prob, A1_prob, b1_prob, x0_0_prob, x0_1_prob, tol, max_iter=10000):
            """
            Βελτιστοποιημένη Επίλυση δύο συστημάτων γραμμικών εξισώσεων με αραιούς πίνακες.
            """
            # Μετατροπή των A0 και A1 σε sparse μορφή
            A0_prob = csr_matrix(A0_prob) if not isinstance(A0_prob, csr_matrix) else A0_prob
            A1_prob = csr_matrix(A1_prob) if not isinstance(A1_prob, csr_matrix) else A1_prob

            n = len(b0_prob)

            # Αρχικές τιμές
            x0_prob = np.zeros(n) if x0_0_prob is None else x0_0_prob
            x1_prob = np.zeros(n) if x0_1_prob is None else x0_1_prob

            diag_A0 = A0_prob.diagonal()  # Διαγώνιος από sparse matrix
            diag_A1 = A1_prob.diagonal()

            for k in range(max_iter):
                # Υπολογισμός του x0_new_prob
                residual_0 = b0_prob - A0_prob.dot(x0_prob)
                x0_new_prob = residual_0 / diag_A0 + x0_prob
                np.maximum(x0_new_prob, x1_prob - k0, out=x0_new_prob)

                # Υπολογισμός του x1_new_prob
                residual_1 = b1_prob - A1_prob.dot(x1_prob)
                x1_new_prob = residual_1 / diag_A1 + x1_prob
                np.maximum(x1_new_prob, x0_new_prob - k1, out=x1_new_prob)

                # Έλεγχος σύγκλισης κάθε 1000 επαναλήψεις
                if k % 100000 == 0:
                    diff0 = np.linalg.norm(x0_new_prob - x0_prob, ord=np.inf)
                    diff1 = np.linalg.norm(x1_new_prob - x1_prob, ord=np.inf)
                    if diff0 < tol and diff1 < tol:
                        return x0_new_prob, x1_new_prob, k + 1, diff0, diff1
                    print(f"Iteration {k}: diff0={diff0:.6e}, diff1={diff1:.6e}")
        
                # Ενημέρωση για την επόμενη επανάληψη
                x0_prob[:] = x0_new_prob
                x1_prob[:] = x1_new_prob

            raise ValueError("Τα συστήματα δεν συγκλίνουν εντός του μέγιστου αριθμού επαναλήψεων.")

        ###### ########### ########### ######### ######### ###### ########### ########### ######### #########
        ###### ########### ########### ######### ######### ###### ########### ########### ######### #########

        # Παράδειγμα χρήσης
        if __name__ == "__main__":
            # Πρώτο σύστημα (prob 0)
            A0_prob = T0_n
            b0_prob = b0_n

            # Δεύτερο σύστημα (prob 1)
            A1_prob = T1_n
            b1_prob = b1_n
            # Ξεκινάμε τη μέτρηση
            start_time = time.time()
            try:
                solution0_prob, solution1_prob, iterations, apo0,apo1 = solve_two_same_size_systems(
                    A0_prob, b0_prob, A1_prob, b1_prob, x0_start, x1_start,1e-8, max_iter=100000000
                )
            except ValueError as e:
                print(e)

            # Υπολογισμός του χρόνου εκτέλεσης
            end_time = time.time()
            execution_time_seconds = end_time - start_time
            execution_time_minutes = execution_time_seconds / 60

            print(f"Χρόνος εκτέλεσης: {execution_time_minutes:.2f} λεπτά")

        ###### ########### ########### ######### ######### ###### ########### ########### ######### #########
        ###### ########### ########### ######### ######### ###### ########### ########### ######### #########
        x0_start=solution0_prob
        x1_start=solution1_prob

        x0=solution0_prob
        x1=solution1_prob

        V0=np.zeros((N+1,M+1))
        V1=np.zeros((N+1,M+1))

        for i in range((N-1)*(M+1)):
            pii,yii=findpy_n(i)
            V0[pii+1,yii]=np.copy(x0[i])
            V1[pii+1,yii]=np.copy(x1[i])

        for i in range(M+1):
            V0[0,i]=V0katwd[i]
            V1[0,i]=V1katwd[i]
        for i in range(M+1):
            V0[N,i]=V0panwd[i]
            V1[N,i]=V1panwd[i]

        V0save=np.copy(V0)
        V1save=np.copy(V1)
        l0save=np.copy(x0)
        l1save=np.copy(x1)

        V0_00=V0
        V1_00=V1


        tableV0=V0.T[2,:]
        tableV1=V1.T[2,:]
        difference01=tableV0-tableV1
        difference10=tableV1-tableV0

        countance = np.arange(tableV1.shape[0])

        plt.plot(countance,difference10, label = "line 1")
        plt.plot(countance,difference01, label = "line 2")


        df_1=pd.DataFrame(V0)
        df_2=pd.DataFrame(V1)
        df_3=pd.DataFrame(V1-V0)

        #with pd.ExcelWriter(r'0001 60 30 6-0.xlsx') as writer1:
        #    df_1.to_excel(writer1, sheet_name = 'df_1', index = False)
        #    df_2.to_excel(writer1, sheet_name = 'df_2', index = False)
        #    df_3.to_excel(writer1, sheet_name = 'df_3', index = False)

        def process_and_merge_arrays_to_dataframe_start(array1, var1, var2,start_col, end_col):
            # Δημιουργία νέας στήλης για τον πρώτο πίνακα
            new_col1 = np.linspace(var1, var2, array1.shape[0]).reshape(-1, 1)  # Δημιουργία στήλης με var1 έως var2
            new_col1 = np.round(new_col1, 5)  # Στρογγυλοποίηση σε 3 δεκαδικά ψηφία
            array2 = np.round(array1, 5)
            new_array1 = np.hstack((new_col1, array2))  # Συνένωση της νέας στήλης με τον πρώτο πίνακα

            # Υπολογισμός ονομάτων στηλών βάσει του range
            column_names = ['P\\Y'] + [f"{round(i, 3)}" for i in np.linspace(start_col, end_col, array1.shape[1])]

            # Μετατροπή σε Pandas DataFrame
            final_dataframe = pd.DataFrame(new_array1, columns=column_names)
    
            return final_dataframe

        Final_result=process_and_merge_arrays_to_dataframe_start(V1-V0,ppmin,ppmax,yymin,yymax)

        # Στρογγυλοποίηση στον 9ο δεκαδικό
        Final_result = Final_result.round(9)

        Final_result

        df = pd.DataFrame(Final_result)



        # Λίστες για αποθήκευση τιμών
        y_values = []
        to_4_p_values = []
        from_minus2_p_values = []

        # Διατρέχουμε κάθε στήλη (εκτός από την πρώτη που είναι το P)
        for col_idx, col_name in enumerate(df.columns[1:], start=1):
            column_values = df.iloc[:, col_idx]
            previous_values = column_values.shift(1)

            # Προσθήκη τιμής Y
            y_values.append(float(col_name))  # Το όνομα της στήλης

           # ✅ Μετάβαση ΠΡΟΣ το 4 (μπαίνει στο 4)
            mask_to_4 = (column_values == 4) & (previous_values != 4)
            if mask_to_4.any():
                first_idx_to_4 = mask_to_4.idxmax()
                p_value_to_4 = df.iloc[first_idx_to_4, 0]
            else:
                p_value_to_4 = np.nan  # Αν δεν υπάρχει, βάζουμε NaN

            to_4_p_values.append(p_value_to_4)

            # ✅ Μετάβαση ΑΠΟ το -2 (φεύγει από -2)
            mask_from_minus2 = (previous_values == -2) & (~column_values.isin([-2]))
            if mask_from_minus2.any():
                first_idx_from_minus2 = mask_from_minus2.idxmax()
                p_value_from_minus2 = df.iloc[first_idx_from_minus2, 0]
            else:
                p_value_from_minus2 = np.nan

            from_minus2_p_values.append(p_value_from_minus2)

        # Στρογγυλοποίηση των P0best και P1best με 5 δεκαδικά
        P0best_value = round(float(P0best), 5)
        P1best_value = round(float(P1best), 5)

        # Προσθέτουμε την τιμή 'analytic' και τα δύο best values στην αρχή της λίστας
        y_values_with_label = y_values + ["analytic"]
        to_4_p_values_with_best = to_4_p_values + [P0best_value]
        from_minus2_p_values_with_best = from_minus2_p_values + [P1best_value]

        # Δημιουργία νέου πίνακα
        final_array = np.array([y_values_with_label, to_4_p_values_with_best, from_minus2_p_values_with_best])

        # Δημιουργία DataFrame
        final_df = pd.DataFrame(final_array, index=["Y", "P (to 4)", "P (from -2)"])


        # --- Στρογγυλοποίηση βοηθητική ---
        def r5(x): 
           return round(float(x), 5)

        # Μεταβλητές και οι τιμές τους
        var_names = ['r', 'rho', 'm', 'd', 'v', 'mi', 'k0', 'k1', 'c']
        var_values = [r, rho, m, d, v, mi, k0, k1, c]
        var_values_rounded = [r5(val) for val in var_values]

        # Μετατροπή των σειρών σε λίστες (για προσθήκη δεδομένων)
        Y_row = list(final_df.loc["Y"])
        P_to_4_row = list(final_df.loc["P (to 4)"])
        P_from_minus2_row = list(final_df.loc["P (from -2)"])

        # Προσθήκη των νέων στοιχείων
        Y_row.extend(var_names)
        P_to_4_row.extend(var_values_rounded)
        P_from_minus2_row.extend(var_values_rounded)

        # Δημιουργία νέου DataFrame
        final_df_extended_pre = pd.DataFrame([Y_row, P_to_4_row, P_from_minus2_row],
                                     index=["Y", "P (to 4)", "P (from -2)"])

        print(final_df_extended_pre)





        def build_tables_from_final_df(final_df_extended):
            """
            Δημιουργεί δύο πίνακες:
            - table_to_4: με πρώτη γραμμή τη Y και δεύτερη την P (to 4)
            - table_from_minus2: με πρώτη γραμμή τη Y και δεύτερη την P (from -2)
            Κοινή δομή: index = [‘Main’, 0], στήλη ‘Y’ + υπόλοιπες Y τιμές
            """
            # Πάρε σειρές
            y_row = final_df_extended.loc["Y"]
            to_4_row = final_df_extended.loc["P (to 4)"]
            from_minus2_row = final_df_extended.loc["P (from -2)"]

            # Δημιουργούμε τις σειρές που θα μπουν στο DataFrame
            y_full_row = ["Y"] + list(y_row)
            to_4_full_row = ["P (to 4)"] + list(to_4_row)
            from_minus2_full_row = ["P (from -2)"] + list(from_minus2_row)

            # Δημιουργούμε τους πίνακες
            table_to_4 = pd.DataFrame([y_full_row, to_4_full_row], index=[0, 1])
            table_from_minus2 = pd.DataFrame([y_full_row, from_minus2_full_row], index=[0, 1])

            return table_to_4, table_from_minus2

        def append_final_df_to_tables(table_to_4, table_from_minus2, final_df_extended_new):
            """
            Παίρνει έναν νέο πίνακα final_df_extended_new και κολλάει:
            - τη γραμμή P (to 4) στον table_to_4
            - τη γραμμή P (from -2) στον table_from_minus2
            Η γραμμή πηγαίνει στο τέλος κάθε πίνακα (index αυξάνεται).
            """
            # Πάρε σειρές
            y_row = final_df_extended_new.loc["Y"]
            to_4_row = final_df_extended_new.loc["P (to 4)"]
            from_minus2_row = final_df_extended_new.loc["P (from -2)"]

            # Αντιγραφή για ασφάλεια
            table_to_4 = table_to_4.copy()
            table_from_minus2 = table_from_minus2.copy()

            # --- Κόλλα στο to_4 ---
            new_index_to_4 = table_to_4.index.max() + 1
            row_to_4 = ["P (to 4)"] + list(to_4_row)
            table_to_4.loc[new_index_to_4] = row_to_4

            # --- Κόλλα στο from_minus2 ---
            new_index_from_2 = table_from_minus2.index.max() + 1
            row_from_minus2 = ["P (from -2)"] + list(from_minus2_row)
            table_from_minus2.loc[new_index_from_2] = row_from_minus2

            return table_to_4, table_from_minus2


        if iterate==0:
            table_to_4_final, table_from_minus2_final = build_tables_from_final_df(final_df_extended_pre)
            iterate=1
        else:
            table_to_4_final, table_from_minus2_final = append_final_df_to_tables(table_to_4_final, table_from_minus2_final, final_df_extended_pre)
            iterate+=1

        elapsed = time.time() - overall_start_time
        elapsed_min = elapsed / 60

        print('✅ We are now at',iterate,'out of',len(combinations),'. Percentage:',100*iterate/len(combinations),'Average Run:',elapsed_min/iterate,'Run:',elapsed_min)

        # ✅ Αποθήκευση κάθε 10 φορές
        if i % 10 == 0:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                table_to_4_final.to_excel(writer, sheet_name="P_to_4", index=True, header=False)
                table_from_minus2_final.to_excel(writer, sheet_name="P_from_minus2", index=True, header=False)
            print(f"✅ Excel updated at i = {i}")

        print("== Πίνακας με P (to 4) ==")
        print(table_to_4_final)

        print("== Πίνακας με P (from -2) ==")
        print(table_from_minus2_final)

# 🌐 Flask app για κατέβασμα
app = Flask(__name__)

@app.route("/")
def home():
    return "<h3>✅ Το πρόγραμμα τρέχει.<br><a href='/download'>Κατέβασε τον πιο πρόσφατο πίνακα (.xlsx)</a></h3>"

@app.route("/download")
def download_excel():
    return send_file(filename, as_attachment=True)

# 🚀 Εκκίνηση loop σε background thread ώστε Flask να είναι live
threading.Thread(target=run_all_combinations, daemon=True).start()

# ✅ Εκκίνηση Flask server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=10000)


# In[ ]:





# In[ ]:




