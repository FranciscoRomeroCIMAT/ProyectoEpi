# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:39:32 2022

@author: migue
"""

import scipy.integrate as spi
import numpy as np
import pylab as pl
import scipy.optimize as optimize
import scipy as sci
import scipy.stats
from statsmodels.graphics.tsaplots import plot_acf
N=1e6
np.random.seed(21)#Semilla
def ode_SEIR(INP,t,beta,gamma,eta,kappa,f,p):  
    """Función que define a la ecuación diferencial, Y[5] corresponde a los nuevos casos por 
    unidad de tiempo de infectados sintomáticos (incidencia de sintomáticos)"""
    Y = np.zeros((6))
    V = INP    
    Y[0] = - beta *( V[3]+eta*V[2])* V[0]/N  
    Y[1] = beta *( V[3]+eta*V[2])* V[0]/N  - kappa * V[1] 
    Y[2] = p*kappa*V[1]-gamma*V[2]
    Y[3] = (1-p)*kappa*V[1]-(gamma+f)*V[3]
    Y[4] = gamma*(V[2]+V[3])+f*V[3]
    Y[5]= (1-p)*kappa*V[1]
    return Y  
#PARAMETROS
beta = 1.42
gamma = 0.22
eta=0.142
kappa=0.833
p=0.333
f=0.5
#Condiciones inciales
S0 = N-1
I0 = 1.
INPUT = (S0,0.,0., I0, 0.0,0.)
#Valor de R0
R0=(beta*eta*p/gamma)+beta*((1-p)/(f+gamma))
print("El valor de R0 es:",R0)
#Tiempo, desde el día 0 hasta el día 200 en unidades de un día
t_start = 0.0; 
t_end = 200; 
t_inc = 1.0
t_range = np.arange(t_start, t_end+t_inc, t_inc)
#Solución numerica mediante odeint
SOL = spi.odeint(ode_SEIR,INPUT,t_range,args=(beta,gamma,eta,kappa,f,p))

#Grafica de S(t) y R(t)
pl.figure()
pl.plot(SOL[:,0], '-g', label='Susc ind')
pl.plot(SOL[:,4], '-k', label='Recov ind')
pl.legend(loc=0)
pl.xlabel('Time')
pl.ylabel('S and R')

#Grafica de I(t)
pl.figure()
pl.plot(SOL[:,3], '-r', label='Infec indiv')
pl.legend(loc=0)
pl.xlabel('Time')
pl.ylabel('I(t)')
pl.show()


solruido=SOL+np.random.normal(size=SOL.shape,scale=1000) #SOL + ruido normal
solruido2=np.random.poisson(SOL[:,3],SOL[:,3].shape) #Variable Poisson con media I(t) para los valores de t
solruido=abs(solruido) #tomamos el valor absoluto de I(t) con ruido normal
pl.figure()
#Graficas de I(t) con y sin ruido
pl.plot(solruido[:,3], '-r', label='Ruido Normal')
pl.plot(solruido2, '-b', label='Ruido Poisson' )
pl.plot(SOL[:,3],'-g',label="Solución sin ruido")
pl.legend(loc=0)
pl.xlabel('Día')
pl.ylabel('I(t)')
pl.show()
#Calculo de incidencia para sintomáticos por día
incidencia=np.diff(SOL[:,5])
#Agregamos ruido normal a la incidencia 
solruido3=abs(incidencia+np.random.normal(size=incidencia.shape,scale=700,))
#Simulamos la incidencia con ruido Poisson
solruido2=np.random.poisson(incidencia,incidencia.shape)
#Agregamos la condición incial al vector de incidencias ya que por defecto no lo incluye
solincdias=np.concatenate((np.array([I0]),solruido3))
solincdias2=np.concatenate((np.array([I0]),solruido2))
SOLincdias=np.concatenate((np.array([I0]),incidencia))
#Graficas de nuevos casos diario con y sin ruido
pl.figure()
pl.plot(solincdias, '-r', label='Ruido Normal')
pl.plot(solincdias2, '-b', label='Ruido Poisson' )
pl.plot(SOLincdias,'-g',label="Solución sin ruido")
pl.legend(loc=0)
pl.xlabel('Día')
pl.ylabel('Nuevos casos diarios')
pl.show()

######
def loss_funPoisson(params): 
    """Logaritmo de la función de perdida cuadratica para los datos de incidencia
    con ruido Poisson para el vector de parametros params"""
    beta,gamma,eta,kappa,f,p = params
    SOL = spi.odeint(ode_SEIR,INPUT,t_range,args=(beta,gamma,eta,kappa,f,p))
    incidencia=np.diff(SOL[:,5])
    SOLincdias=np.concatenate((np.array([I0]),incidencia))
    loss = np.log(np.sum((SOLincdias-solincdias2)**2)/200)
    return np.array([loss])
def loss_funNorm(params): 
    """Logaritmo de la función de perdida cuadratica para los datos de incidencia
    con ruido Normal para el vector de parametros params"""
    beta,gamma,eta,kappa,f,p = params
    SOL = spi.odeint(ode_SEIR,INPUT,t_range,args=(beta,gamma,eta,kappa,f,p))
    incidencia=np.diff(SOL[:,5])
    SOLincdias=np.concatenate((np.array([I0]),incidencia))
    loss = np.log(np.sum((SOLincdias-solincdias)**2)/200)
    return np.array([loss])

#Estimaciones inciales de los parámetros
initial_guess=np.array([1.1,0.2,0.4,0.2,0.3,0.2]) 
#Le decimos los intervalos donde busco los valores que minimizan
bnds= ((0.0000001, 3), (0.001, 1),(0.001, 1),(0.001, 1),(0.001, 1),(0.001, 1))
result = optimize.minimize(loss_funPoisson,initial_guess,bounds=bnds,method='Nelder-Mead')
paramPois=result.x #parametros estimados para incidencias de ruido Poisson
SOLPoiss = spi.odeint(ode_SEIR,INPUT,t_range,args=tuple(paramPois)) #Solución con parametros estimados
IncajustePois=np.concatenate((np.array([I0]),np.diff(SOLPoiss[:,5])))#Incidencia con parametros estimados
print("Parametros Ajuste Poisson:",paramPois) 
(beta_p,gamma_p,eta_p,kappa_p,f_p,p_p)=paramPois
R0_p=(beta_p*eta_p*p_p/gamma_p)+beta_p*((1-p_p)/(f_p+gamma_p)) #R0 para parametros estimados
print("R0 Ajuste Poisson:",R0_p)

##########
result = optimize.minimize(loss_funNorm,initial_guess,bounds=bnds,method='Nelder-Mead')
paramNorm=result.x #parametros estimados para incidencias de ruido Normal
SOLNorm = spi.odeint(ode_SEIR,INPUT,t_range,args=tuple(paramNorm))#Solución con parametros estimados
IncajusteNorm=np.concatenate((np.array([I0]),np.diff(SOLNorm[:,5])))#Incidencia con parametros estimados
print("Parametros Ajuste Normal:",paramNorm)
(beta_n,gamma_n,eta_n,kappa_n,f_n,p_n)=paramNorm
R0_n=(beta_n*eta_n*p_n/gamma_n)+beta_n*((1-p_n)/(f_n+gamma_n))#R0 para parametros estimados
print("R0 Ajuste Normal:",R0_n)

####
#Grafica para incidencias con los parámetros estimados, y la incidencia real
pl.figure()
pl.plot(IncajusteNorm, '-r', label='Ajuste con Ruido Normal ')
pl.plot(IncajustePois, '-b', label='Ajuste con Ruido Poisson' )
pl.plot(SOLincdias,'-g',label="Solución sin ruido")
pl.legend(loc=0)
pl.xlabel('Día')
pl.ylabel('Nuevos casos diarios')
pl.show()
#Grafica para incidencias con ruido normal, y su curva de incidencia de los parametros estimados
pl.figure()
pl.plot(IncajusteNorm, '-r', label='Ajuste con Ruido Normal')
pl.plot(solincdias, '-b', label='Ruido Normal' )
pl.legend(loc=0)
pl.xlabel('Día')
pl.ylabel('Nuevos casos diarios')
pl.show()
#Grafica para incidencias con ruido normal, y su curva de incidencia de los parametros estimados
pl.figure()
pl.plot(IncajustePois, '-r', label='Ajuste con Ruido Poisson')
pl.plot(solincdias2, '-b', label='Ruido Poisson' )
pl.legend(loc=0)
pl.xlabel('Día')
pl.ylabel('Nuevos casos diarios')
pl.show()
##################### INFERENCIA BAYESIANA
Datos=solruido2
"""
INTENTO DE USAR A PRIORIS INFORMATIVAS - NO JALA :(
def LogKernelGamma(x,a,b):
    out=(a-1)*np.log(x)-b*x
    return(out)

def logvero(beta,gamma_2,eta,kappa,p,f):
    param=(beta,gamma_2,eta,kappa,f,p)
    SolED=spi.odeint(ode_SEIR,INPUT,t_range,args=tuple(param))
    IncD=np.diff(SolED[:,5])
    LV=0
    for i in range(0,len(IncD)):
        if IncD[i]>0:
            LV=LV-IncD[i]+Datos[i]*np.log(IncD[i])
    #LV=np.sum(sci.stats.poisson.logpmf(Datos,IncD))
    LV=LV+LogKernelGamma(beta,2,1)+LogKernelGamma(eta,1,0.1)+LogKernelGamma(kappa,2,0.1)+LogKernelGamma(gamma_2,2,0.1)+LogKernelGamma(f,2,0.1)
    #LV=LV+sci.stats.gamma.logpdf(beta,a=2,loc=0,scale=1)+sci.stats.uniform.logpdf(gamma_2,loc=0,scale=1)
    #LV=LV+sci.stats.gamma.logpdf(eta,a=1,loc=0,scale=.1)+sci.stats.gamma.logpdf(kappa,a=2,loc=0,scale=.1)
    return(LV)
def logvero_f(beta,gamma,eta,kappa,p,f):
    param=(beta,gamma,eta,kappa,f,p)
    SolED=spi.odeint(ode_SEIR,INPUT,t_range,args=tuple(param))
    IncD=np.diff(SolED[:,5])
    LV=0
    for i in range(0,len(IncD)):
        if IncD[i]>0:
            LV=LV-IncD[i]+Datos[i]*np.log(IncD[i])
    #LV=np.sum(sci.stats.poisson.logpmf(Datos,IncD))
    #LV=LV+sci.stats.gamma.logpdf(beta,a=2,loc=0,scale=1)+sci.stats.uniform.logpdf(gamma,loc=0,scale=1)
    return(LV)
def MH_Pois (M,param_inicial,w):
    beta_sim=np.zeros(M)
    gamma_sim=np.zeros(M)
    eta_sim=np.zeros(M)
    kappa_sim=np.zeros(M)
    p_sim=np.zeros(M)
    f_sim=np.zeros(M)
    Vero=np.zeros(M)
    beta_sim[0]=param_inicial[0]
    gamma_sim[0]=param_inicial[1]
    eta_sim[0]=param_inicial[2]
    kappa_sim[0]=param_inicial[3]
    p_sim[0]=param_inicial[4]
    f_sim[0]=param_inicial[5]
    Vero[0]=logvero(beta_sim[0],gamma_sim[0],eta_sim[0],kappa_sim[0],p_sim[0],f_sim[0])
    for i in range(1,M):
        U=sci.stats.uniform.rvs()
        if (U<=w[0]):
            #####PRIMERA PROPUESTA Cambiamos beta
            beta_prop=sci.stats.norm.rvs(loc=beta_sim[i-1],scale=0.1)
            gamma_sim[i]=gamma_sim[i-1]
            eta_sim[i]=eta_sim[i-1]
            kappa_sim[i]=kappa_sim[i-1]
            p_sim[i]=p_sim[i-1]
            f_sim[i]=f_sim[i-1]
            if beta_prop<=0:
                #Rechazamos
                beta_sim[i]=beta_sim[i-1]
                Vero[i]=Vero[i-1]
                continue
            u=sci.stats.uniform.rvs()
            lograzon=logvero(beta_prop,gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])-logvero(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1])
            razon=np.exp(lograzon)
            if u<razon:
                #Aceptamos
                beta_sim[i]=beta_prop
                Vero[i]=logvero(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])
                #print(razon)
            else:
                #Rechazamos
                beta_sim[i]=beta_sim[i-1]
                Vero[i]=Vero[i-1]
            continue
        elif(U<=w[0]+w[1]):
            #####SEGUNDA PROPUESTA Cambiamos gamma
            beta_sim[i]=beta_sim[i-1]
            gamma_prop=sci.stats.norm.rvs(loc=gamma_sim[i-1],scale=0.1)
            eta_sim[i]=eta_sim[i-1]
            kappa_sim[i]=kappa_sim[i-1]
            p_sim[i]=p_sim[i-1]
            f_sim[i]=f_sim[i-1]
            if gamma_prop<=0:
                #Rechazamos
                gamma_sim[i]=gamma_sim[i-1]
                Vero[i]=Vero[i-1]
                continue
            u=sci.stats.uniform.rvs()
            lograzon=logvero(beta_sim[i],gamma_prop,eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])-logvero(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1])
            razon=np.exp(lograzon)
            if u<razon:
                #Aceptamos
                gamma_sim[i]=gamma_prop
                Vero[i]=logvero(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])
                #print(razon)
            else:
                #Rechazamos
                gamma_sim[i]=gamma_sim[i-1]
                Vero[i]=Vero[i-1]
            continue
        elif(U<=w[0]+w[1]+w[2]):
            #####TERCER PROPUESTA Cambiamos eta
            beta_sim[i]=beta_sim[i-1]
            gamma_sim[i]=gamma_sim[i-1]
            eta_prop=sci.stats.norm.rvs(loc=eta_sim[i-1],scale=0.1)
            kappa_sim[i]=kappa_sim[i-1]
            p_sim[i]=p_sim[i-1]
            f_sim[i]=f_sim[i-1]
            if eta_prop<=0:
                #Rechazamos
                eta_sim[i]=eta_sim[i-1]
                Vero[i]=Vero[i-1]
                continue
            u=sci.stats.uniform.rvs()
            lograzon=logvero(beta_sim[i],gamma_sim[i],eta_prop,kappa_sim[i],p_sim[i],f_sim[i])-logvero(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1])
            razon=np.exp(lograzon)
            if u<razon:
                #Aceptamos
                eta_sim[i]=eta_prop
                Vero[i]=logvero(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])
                #print(razon)
            else:
                #Rechazamos
                eta_sim[i]=eta_sim[i-1]
                Vero[i]=Vero[i-1]
            continue
        elif(U<=w[0]+w[1]+w[2]+w[3]):
            #####CUARTA PROPUESTA Cambiamos kappa
            beta_sim[i]=beta_sim[i-1]
            gamma_sim[i]=gamma_sim[i-1]
            eta_sim[i]=eta_sim[i-1]
            kappa_prop=sci.stats.norm.rvs(loc=kappa_sim[i-1],scale=0.1)
            p_sim[i]=p_sim[i-1]
            f_sim[i]=f_sim[i-1]
            if kappa_prop<=0:
                #Rechazamos
                kappa_sim[i]=kappa_sim[i-1]
                Vero[i]=Vero[i-1]
                continue
            u=sci.stats.uniform.rvs()
            lograzon=logvero(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_prop,p_sim[i],f_sim[i])-logvero(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1])
            razon=np.exp(lograzon)
            if u<razon:
                #Aceptamos
                kappa_sim[i]=kappa_prop
                Vero[i]=logvero(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])
                #print(razon)
            else:
                #Rechazamos
                kappa_sim[i]=kappa_sim[i-1]
                Vero[i]=Vero[i-1]
            continue
        elif(U<=w[0]+w[1]+w[2]+w[3]+w[4]):
            #####QUINTA PROPUESTA Cambiamos p
            beta_sim[i]=beta_sim[i-1]
            gamma_sim[i]=gamma_sim[i-1]
            eta_sim[i]=eta_sim[i-1]
            kappa_sim[i]=kappa_sim[i-1]
            p_prop=sci.stats.norm.rvs(loc=p_sim[i-1],scale=0.1)
            #p_prop=sci.stats.beta.rvs(a=2,b=5)
            f_sim[i]=f_sim[i-1]
            u=sci.stats.uniform.rvs()
            lograzon=logvero(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_prop,f_sim[i])-logvero(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1])
            #lograzon=lograzon+sci.stats.beta.logpdf(p_sim[i-1],a=2,b=5)-sci.stats.beta.logpdf(p_prop,a=2,b=5)
            razon=np.exp(lograzon)
            if (p_prop<=0 or p_prop>=0.5):
                #Rechazamos
                p_sim[i]=p_sim[i-1]
                Vero[i]=Vero[i-1]
                continue
            if u<razon:
                #Aceptamos
                p_sim[i]=p_prop
                Vero[i]=logvero(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])
                #print(razon)
            else:
                #Rechazamos
                p_sim[i]=p_sim[i-1]
                Vero[i]=Vero[i-1]
            continue
        else:
            #####SEXTA PROPUESTA Cambiamos f
            beta_sim[i]=beta_sim[i-1]
            gamma_sim[i]=gamma_sim[i-1]
            eta_sim[i]=eta_sim[i-1]
            kappa_sim[i]=kappa_sim[i-1]
            p_sim[i]=p_sim[i-1]
            f_prop=sci.stats.norm.rvs(loc=f_sim[i-1],scale=0.1)
            if (f_prop<=0 or f_prop>=1):
                #Rechazamos
                f_sim[i]=f_sim[i-1]
                Vero[i]=Vero[i-1]
                continue
            u=sci.stats.uniform.rvs()
            lograzon=logvero_f(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_prop)-logvero_f(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1])
            razon=np.exp(lograzon)
            #print(logvero(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_prop))
            #print("prop:",logvero(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1]))
            if u<razon:
                #Aceptamos
                f_sim[i]=f_prop
                Vero[i]=logvero_f(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])
                #print(razon)
            else:
                #Rechazamos
                f_sim[i]=f_sim[i-1]
                Vero[i]=Vero[i-1]
            continue
    return(beta_sim,gamma_sim,eta_sim,kappa_sim,p_sim,f_sim,Vero)

np.random.seed(2022)
 
beta_inicial=paramPois[0]+sci.stats.norm.rvs()
gamma_inicial=paramPois[1]
eta_inicial=paramPois[2]
kappa_inicial=paramPois[3]
p_inicial=paramPois[5]
f_inicial=paramPois[4]
 
beta_inicial=sci.stats.gamma.rvs(a=1,loc=0,scale=1)
gamma_inicial=sci.stats.gamma.rvs(a=3,loc=0,scale=0.01)
eta_inicial=sci.stats.gamma.rvs(a=1,loc=0,scale=.1)
kappa_inicial=sci.stats.gamma.rvs(a=3,loc=0,scale=0.1)
p_inicial=sci.stats.uniform.rvs(loc=0,scale=0.2)
f_inicial=sci.stats.uniform.rvs(loc=0,scale=0.5)
#f_inicial=sci.stats.gamma.rvs(a=2,loc=0,scale=0.1)

#print(logvero(beta_inicial, gamma_inicial, eta_inicial, kappa_inicial, p_inicial, f_inicial))
param_inicial=np.array([beta_inicial,gamma_inicial,eta_inicial,kappa_inicial,p_inicial,f_inicial])
M=50000
w=np.array([1/6,1/6,1/6,1/6,1/6,1/6])
beta_sim,gamma_sim,eta_sim,kappa_sim,p_sim,f_sim,Vero=MH_Pois(M,param_inicial,w)
pl.figure()
pl.plot(Vero[10000:])
pl.title("Logverosimilitud")
pl.figure()
pl.plot(beta_sim[10000:])
pl.title("Beta simulada")
pl.figure()
pl.plot(gamma_sim[10000:])
pl.title("Gamma simulada")
pl.figure()
pl.plot(eta_sim[10000:])
pl.title("eta simulada")
pl.figure()
pl.plot(kappa_sim[10000:])
pl.title("Kappa simulada")
pl.figure()
pl.plot(p_sim[10000:])
pl.title("p simulada")
pl.figure()
pl.plot(f_sim[10000:])
pl.title("f simulada")

plot_acf(beta_sim[10000:],lags=600,title="ACF Beta")
plot_acf(gamma_sim[10000:],lags=600,title="ACF Gamma")
plot_acf(eta_sim[10000:],lags=600,title="ACF Eta")
plot_acf(kappa_sim[10000:],lags=600,title="ACF Kappa")
plot_acf(p_sim[10000:],lags=600,title="ACF p")
plot_acf(f_sim[10000:],lags=600,title="ACF f")

beta_fin=beta_sim[range(10000,M,300)]
gamma_fin=gamma_sim[range(10000,M,300)]
eta_fin=eta_sim[range(10000,M,300)]
kappa_fin=kappa_sim[range(10000,M,300)]
p_fin=p_sim[range(10000,M,300)]
f_fin=f_sim[range(10000,M,300)]

beta_b=np.mean(beta_fin)
gamma_b=np.mean(gamma_fin)
eta_b=np.mean(eta_fin)
kappa_b=np.mean(kappa_fin)
p_b=np.mean(p_fin)
f_b=np.mean(f_fin)
param=(beta_b,gamma_b,eta_b,kappa_b,f_b,p_b)
SolED=spi.odeint(ode_SEIR,INPUT,t_range,args=tuple(param))
IncD=np.diff(SolED[:,5])
pl.figure()
pl.plot(IncD, '-r', label='Ajuste con Bayesiana ')
pl.plot(Datos, '-b', label='Datos con ruido Poisson' )

"""
######################
def logvero_noInf(beta,gamma_2,eta,kappa,p,f):
    param=(beta,gamma_2,eta,kappa,f,p)
    SolED=spi.odeint(ode_SEIR,INPUT,t_range,args=tuple(param))
    IncD=np.diff(SolED[:,5])
    LV=0
    for i in range(0,len(IncD)):
        if IncD[i]>0:
            LV=LV-IncD[i]+Datos[i]*np.log(IncD[i])
    #LV=np.sum(sci.stats.poisson.logpmf(Datos,IncD))
    #LV=LV+LogKernelGamma(beta,2,1)+LogKernelGamma(eta,1,0.1)+LogKernelGamma(kappa,2,0.1)+LogKernelGamma(gamma_2,2,0.1)+LogKernelGamma(f,2,0.1)
    #LV=LV+sci.stats.gamma.logpdf(beta,a=2,loc=0,scale=1)+sci.stats.uniform.logpdf(gamma_2,loc=0,scale=1)
    #LV=LV+sci.stats.gamma.logpdf(eta,a=1,loc=0,scale=.1)+sci.stats.gamma.logpdf(kappa,a=2,loc=0,scale=.1)
    return(LV)

def MH_PoisNoInf (M,param_inicial,w):
    beta_sim=np.zeros(M)
    gamma_sim=np.zeros(M)
    eta_sim=np.zeros(M)
    kappa_sim=np.zeros(M)
    p_sim=np.zeros(M)
    f_sim=np.zeros(M)
    Vero=np.zeros(M)
    beta_sim[0]=param_inicial[0]
    gamma_sim[0]=param_inicial[1]
    eta_sim[0]=param_inicial[2]
    kappa_sim[0]=param_inicial[3]
    p_sim[0]=param_inicial[4]
    f_sim[0]=param_inicial[5]
    Vero[0]=logvero_noInf(beta_sim[0],gamma_sim[0],eta_sim[0],kappa_sim[0],p_sim[0],f_sim[0])
    for i in range(1,M):
        #print("i-1:",i-1," y beta:",beta_sim[i-1])
        U=sci.stats.uniform.rvs()
        if (U<=w[0]):
            #####PRIMERA PROPUESTA Cambiamos beta
            beta_prop=sci.stats.norm.rvs(loc=beta_sim[i-1],scale=0.1)
            gamma_sim[i]=gamma_sim[i-1]
            eta_sim[i]=eta_sim[i-1]
            kappa_sim[i]=kappa_sim[i-1]
            p_sim[i]=p_sim[i-1]
            f_sim[i]=f_sim[i-1]
            if (beta_prop>0 and beta_prop<3):
                #soporte de Unif(0,3) de apriori
                u=sci.stats.uniform.rvs()
                lograzon=logvero_noInf(beta_prop,gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])-logvero_noInf(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1])
                razon=np.exp(lograzon)
                if u<razon:
                    #Aceptamos
                    beta_sim[i]=beta_prop
                    #print("aceptado",beta_sim[i])
                    Vero[i]=logvero_noInf(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])
                    #print(razon)
                else:
                    #Rechazamos
                    beta_sim[i]=beta_sim[i-1]
                    #print("rechazado",beta_sim[i])
                    Vero[i]=Vero[i-1]
                continue
            #Rechazamos
            beta_sim[i]=beta_sim[i-1]
            Vero[i]=Vero[i-1]
            continue
        elif(U<=w[0]+w[1]):
            #####SEGUNDA PROPUESTA Cambiamos gamma
            beta_sim[i]=beta_sim[i-1]
            gamma_prop=sci.stats.norm.rvs(loc=gamma_sim[i-1],scale=0.01)
            eta_sim[i]=eta_sim[i-1]
            kappa_sim[i]=kappa_sim[i-1]
            p_sim[i]=p_sim[i-1]
            f_sim[i]=f_sim[i-1]
            if (gamma_prop<=0 or gamma_prop>=1):
                #soporte de Unif(0,1) de apriori
                #Rechazamos
                gamma_sim[i]=gamma_sim[i-1]
                Vero[i]=Vero[i-1]
                continue
            u=sci.stats.uniform.rvs()
            lograzon=logvero_noInf(beta_sim[i],gamma_prop,eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])-logvero_noInf(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1])
            razon=np.exp(lograzon)
            if u<razon:
                #Aceptamos
                gamma_sim[i]=gamma_prop
                Vero[i]=logvero_noInf(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])
                #print(razon)
            else:
                #Rechazamos
                gamma_sim[i]=gamma_sim[i-1]
                Vero[i]=Vero[i-1]
            continue
        elif(U<=w[0]+w[1]+w[2]):
            #####TERCER PROPUESTA Cambiamos eta
            beta_sim[i]=beta_sim[i-1]
            gamma_sim[i]=gamma_sim[i-1]
            eta_prop=sci.stats.norm.rvs(loc=eta_sim[i-1],scale=0.01)
            kappa_sim[i]=kappa_sim[i-1]
            p_sim[i]=p_sim[i-1]
            f_sim[i]=f_sim[i-1]
            if (eta_prop<=0 or eta_prop>=1):
                #Rechazamos #soporte de Unif(0,1) de apriori
                eta_sim[i]=eta_sim[i-1]
                Vero[i]=Vero[i-1]
                continue
            u=sci.stats.uniform.rvs()
            lograzon=logvero_noInf(beta_sim[i],gamma_sim[i],eta_prop,kappa_sim[i],p_sim[i],f_sim[i])-logvero_noInf(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1])
            razon=np.exp(lograzon)
            if u<razon:
                #Aceptamos
                eta_sim[i]=eta_prop
                Vero[i]=logvero_noInf(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])
                #print(razon)
            else:
                #Rechazamos
                eta_sim[i]=eta_sim[i-1]
                Vero[i]=Vero[i-1]
            continue
        elif(U<=w[0]+w[1]+w[2]+w[3]):
            #####CUARTA PROPUESTA Cambiamos kappa
            beta_sim[i]=beta_sim[i-1]
            gamma_sim[i]=gamma_sim[i-1]
            eta_sim[i]=eta_sim[i-1]
            kappa_prop=sci.stats.norm.rvs(loc=kappa_sim[i-1],scale=0.01)
            p_sim[i]=p_sim[i-1]
            f_sim[i]=f_sim[i-1]
            if (kappa_prop<=0 or kappa_prop>=1):
                #Rechazamos
                #soporte de Unif(0,1) de apriori
                kappa_sim[i]=kappa_sim[i-1]
                Vero[i]=Vero[i-1]
                continue
            u=sci.stats.uniform.rvs()
            lograzon=logvero_noInf(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_prop,p_sim[i],f_sim[i])-logvero_noInf(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1])
            razon=np.exp(lograzon)
            if u<razon:
                #Aceptamos
                kappa_sim[i]=kappa_prop
                Vero[i]=logvero_noInf(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])
                #print(razon)
            else:
                #Rechazamos
                kappa_sim[i]=kappa_sim[i-1]
                Vero[i]=Vero[i-1]
            continue
        elif(U<=w[0]+w[1]+w[2]+w[3]+w[4]):
            #####QUINTA PROPUESTA Cambiamos p
            beta_sim[i]=beta_sim[i-1]
            gamma_sim[i]=gamma_sim[i-1]
            eta_sim[i]=eta_sim[i-1]
            kappa_sim[i]=kappa_sim[i-1]
            p_prop=sci.stats.norm.rvs(loc=p_sim[i-1],scale=0.01)
            #p_prop=sci.stats.beta.rvs(a=2,b=5)
            f_sim[i]=f_sim[i-1]
            u=sci.stats.uniform.rvs()
            lograzon=logvero_noInf(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_prop,f_sim[i])-logvero_noInf(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1])
            #lograzon=lograzon+sci.stats.beta.logpdf(p_sim[i-1],a=2,b=5)-sci.stats.beta.logpdf(p_prop,a=2,b=5)
            razon=np.exp(lograzon)
            if (p_prop<=0 or p_prop>=0.5):
                #Rechazamos
                #soporte de Unif(0,0.5) de apriori
                p_sim[i]=p_sim[i-1]
                Vero[i]=Vero[i-1]
                continue
            if u<razon:
                #Aceptamos
                p_sim[i]=p_prop
                Vero[i]=logvero_noInf(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])
                #print(razon)
            else:
                #Rechazamos
                p_sim[i]=p_sim[i-1]
                Vero[i]=Vero[i-1]
            continue
        else:
            #####SEXTA PROPUESTA Cambiamos f
            beta_sim[i]=beta_sim[i-1]
            gamma_sim[i]=gamma_sim[i-1]
            eta_sim[i]=eta_sim[i-1]
            kappa_sim[i]=kappa_sim[i-1]
            p_sim[i]=p_sim[i-1]
            f_prop=sci.stats.norm.rvs(loc=f_sim[i-1],scale=0.1)
            if (f_prop<=0 or f_prop>=1):
                #Rechazamos
                #soporte de Unif(0,1) de apriori
                f_sim[i]=f_sim[i-1]
                Vero[i]=Vero[i-1]
                continue
            u=sci.stats.uniform.rvs()
            lograzon=logvero_noInf(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_prop)-logvero_noInf(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1])
            razon=np.exp(lograzon)
            #print(logvero(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_prop))
            #print("prop:",logvero(beta_sim[i-1],gamma_sim[i-1],eta_sim[i-1],kappa_sim[i-1],p_sim[i-1],f_sim[i-1]))
            if u<razon:
                #Aceptamos
                f_sim[i]=f_prop
                Vero[i]=logvero_noInf(beta_sim[i],gamma_sim[i],eta_sim[i],kappa_sim[i],p_sim[i],f_sim[i])
                #print(razon)
            else:
                #Rechazamos
                f_sim[i]=f_sim[i-1]
                Vero[i]=Vero[i-1]
            continue
    return(beta_sim,gamma_sim,eta_sim,kappa_sim,p_sim,f_sim,Vero)
"""
beta_inicial=paramPois[0]
gamma_inicial=paramPois[1]
eta_inicial=paramPois[2]
kappa_inicial=paramPois[3]
p_inicial=paramPois[5]
f_inicial=paramPois[4]
"""
np.random.seed(2022)#Con esta semilla los valores iniciales de los parámetros estan
#en el soporte de la posterior

beta_inicial=sci.stats.gamma.rvs(a=2,loc=0,scale=1) 
gamma_inicial=sci.stats.gamma.rvs(a=3,loc=0,scale=0.01)
eta_inicial=sci.stats.gamma.rvs(a=1,loc=0,scale=.1)
kappa_inicial=sci.stats.gamma.rvs(a=3,loc=0,scale=0.1)
p_inicial=sci.stats.uniform.rvs(loc=0,scale=0.2)
f_inicial=sci.stats.uniform.rvs(loc=0,scale=0.5)
#f_inicial=sci.stats.gamma.rvs(a=2,loc=0,scale=0.1)  

#print(logvero(beta_inicial, gamma_inicial, eta_inicial, kappa_inicial, p_inicial, f_inicial))
param_inicial=np.array([beta_inicial,gamma_inicial,eta_inicial,kappa_inicial,p_inicial,f_inicial])
M=150000
w=np.array([1/6,1/6,1/6,1/6,1/6,1/6])
np.random.seed(2022)
beta_sim,gamma_sim,eta_sim,kappa_sim,p_sim,f_sim,Vero=MH_PoisNoInf(M,param_inicial,w)
pl.figure()
pl.plot(Vero[50000:])
pl.title("Logverosimilitud")
pl.figure()
pl.plot(beta_sim[50000:])
pl.title("Beta simulada")
pl.figure()
pl.plot(gamma_sim[50000:])
pl.title("Gamma simulada")
pl.figure()
pl.plot(eta_sim[50000:])
pl.title("eta simulada")
pl.figure()
pl.plot(kappa_sim[50000:])
pl.title("Kappa simulada")
pl.figure()
pl.plot(p_sim[50000:])
pl.title("p simulada")
pl.figure()
pl.plot(f_sim[50000:])
pl.title("f simulada")

plot_acf(beta_sim[50000:],lags=600,title="ACF Beta")
plot_acf(gamma_sim[50000:],lags=600,title="ACF Gamma")
plot_acf(eta_sim[50000:],lags=600,title="ACF Eta")
plot_acf(kappa_sim[50000:],lags=600,title="ACF Kappa")
plot_acf(p_sim[50000:],lags=600,title="ACF p")
plot_acf(f_sim[50000:],lags=600,title="ACF f")

beta_fin=beta_sim[range(50000,M,500)]
gamma_fin=gamma_sim[range(50000,M,500)]
eta_fin=eta_sim[range(50000,M,500)]
kappa_fin=kappa_sim[range(50000,M,500)]
p_fin=p_sim[range(50000,M,500)]
f_fin=f_sim[range(50000,M,500)]

beta_b=np.median(beta_fin)
gamma_b=np.median(gamma_fin)
eta_b=np.median(eta_fin)
kappa_b=np.median(kappa_fin)
p_b=np.median(p_fin)
f_b=np.median(f_fin)
param=(beta_b,gamma_b,eta_b,kappa_b,f_b,p_b)
SolED=spi.odeint(ode_SEIR,INPUT,t_range,args=tuple(param))
IncD=np.diff(SolED[:,5])
pl.figure()
pl.plot(IncD, '-r', label='Ajuste con Bayesiana ')
pl.plot(Datos, '-b', label='Datos con ruido Poisson' )
pl.title("Incidencia diaria")
pl.legend(loc="upper right")
#
print("tamaño de muestra final:",len(beta_fin)) #200
q1,q2=np.quantile(beta_fin,(0.025,0.975))
print("Intervalo del 95% cuantilico de beta: [",q1,",",q2,"] y estimador :",beta_b)

q1,q2=np.quantile(gamma_fin,(0.025,0.975))
print("Intervalo del 95% cuantilico de gamma: [",q1,",",q2,"] y estimador :",gamma_b)

q1,q2=np.quantile(eta_fin,(0.025,0.975))
print("Intervalo del 95% cuantilico de eta: [",q1,",",q2,"] y estimador :",eta_b)

q1,q2=np.quantile(kappa_fin,(0.025,0.975))
print("Intervalo del 95% cuantilico de kappa: [",q1,",",q2,"] y estimador :",kappa_b)

q1,q2=np.quantile(p_fin,(0.025,0.975))
print("Intervalo del 95% cuantilico de p: [",q1,",",q2,"] y estimador :",p_b)

q1,q2=np.quantile(f_fin,(0.025,0.975))
print("Intervalo del 95% cuantilico de f: [",q1,",",q2,"] y estimador :",f_b)

R0=(beta_b*eta_b*p_b/gamma_b)+beta_b*((1-p_b)/(f_b+gamma_b))
print("El valor de R0 estimado es:",R0)