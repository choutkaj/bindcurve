import numpy as np

def cubic_lebedev(LT, RT, LsT, Kds, Kd): 

    a = 1
    b = Kds + Kd + LsT + LT - RT
    c = Kds*(LT - RT) + Kd*(LsT - RT) + Kds*Kd
    d = -Kds*Kd*RT
    
    
    tau = b**2 - 3*a*c
    q = b*(9*a*c - 2*b**2) - 27*(a**2) * d
    
    A = q/(2*tau*np.sqrt(tau))

    sigma = (3*a)**(-1)
    
    phi = np.arccos(A)
    cnull = np.cos(phi/3)
    p = cnull*np.sqrt(tau)
    #r = np.sqrt(3)*np.sqrt(tau)*(np.absolute(1-cnull**2))**0.5
    
    x1 = sigma*(2*p-b)
    #x2 = sigma*(r-p-b)
    #x3 = -sigma*(p+r+b)
    
    R = x1
    
    Fsb = R/(Kds + R)
    
    return Fsb


def cubic_numeric(LT, RT, LsT, Kds, Kd): 

    Fsb_array = []
    for i in LT:
        a = Kds + Kd + LsT + i - RT
        b = Kds*(i - RT) + Kd*(LsT - RT) + Kds*Kd
        c = -Kds*Kd*RT
        
        p = [1, a, b, c] 
        roots = np.roots(p) 
        R = roots[0]
        R = max(roots)
        
        Fsb = R/(Kds + R)
        
        Fsb_array.append(Fsb)
    
    return Fsb_array


def quintic_numeric_Fsb(LT, RT, LsT, Kd1, Kd2): 

    Kd3 = 1
    
    Fsb_array = []
    for LT in LT:
        a = Kd1*LsT**3
        b = Kd1*LsT**2*(-Kd1 - 2*Kd3 - 3*LsT - 2*RT)
        c = LsT*(2*Kd1**2*Kd3 + Kd1**2*LT + 2*Kd1**2*LsT + Kd1**2*RT + Kd1*Kd2*Kd3 + Kd1*Kd3**2 - Kd1*Kd3*LT + 4*Kd1*Kd3*LsT + 3*Kd1*Kd3*RT + 3*Kd1*LsT**2 + 6*Kd1*LsT*RT + Kd1*RT**2 - Kd2*Kd3**2)
        d = -Kd1**2*Kd2*Kd3 - Kd1**2*Kd3**2 - Kd1**2*Kd3*LT - 2*Kd1**2*Kd3*LsT - Kd1**2*Kd3*RT - 2*Kd1**2*LT*LsT - Kd1**2*LT*RT - Kd1**2*LsT**2 - 2*Kd1**2*LsT*RT + Kd1*Kd2*Kd3**2 - 2*Kd1*Kd2*Kd3*LsT - Kd1*Kd2*Kd3*RT + Kd1*Kd3**2*LT - Kd1*Kd3**2*LsT - Kd1*Kd3**2*RT + 2*Kd1*Kd3*LT*LsT + Kd1*Kd3*LT*RT - 2*Kd1*Kd3*LsT**2 - 6*Kd1*Kd3*LsT*RT - Kd1*Kd3*RT**2 - Kd1*LsT**3 - 6*Kd1*LsT**2*RT - 3*Kd1*LsT*RT**2 + 2*Kd2*Kd3**2*LsT + Kd2*Kd3**2*RT
        e = Kd1**2*Kd2*Kd3 + Kd1**2*Kd3*LT + Kd1**2*Kd3*RT + Kd1**2*LT*LsT + 2*Kd1**2*LT*RT + Kd1**2*LsT*RT - Kd1*Kd2*Kd3**2 + Kd1*Kd2*Kd3*LsT + 2*Kd1*Kd2*Kd3*RT - Kd1*Kd3**2*LT + Kd1*Kd3**2*RT - Kd1*Kd3*LT*LsT - 2*Kd1*Kd3*LT*RT + 3*Kd1*Kd3*LsT*RT + 2*Kd1*Kd3*RT**2 + 2*Kd1*LsT**2*RT + 3*Kd1*LsT*RT**2 - Kd2*Kd3**2*LsT - 2*Kd2*Kd3**2*RT
        f = RT*(-Kd1**2*LT - Kd1*Kd2*Kd3 + Kd1*Kd3*LT - Kd1*Kd3*RT - Kd1*LsT*RT + Kd2*Kd3**2)


        p = [a, b, c, d, e, f] 
        roots = np.roots(p) 
        Fsb = roots[4]
        
        Fsb_array.append(Fsb)
    
    return Fsb_array



def quintic_numeric_R(LT, RT, LsT, Kd1, Kd2): 

    Kd3 = 100
    
    Fsb_array = []
    for LT in LT:
        a = -Kd1**2*Kd3**2
        b = Kd1**2*Kd3*(Kd1*Kd2 - 3*Kd1*Kd3 + Kd1*LT - 2*Kd1*LsT + Kd1*RT - Kd2*Kd3 - Kd3*LT - Kd3*LsT + Kd3*RT)
        c = Kd1**2*(3*Kd1**2*Kd2*Kd3 - 3*Kd1**2*Kd3**2 + 3*Kd1**2*Kd3*LT - 4*Kd1**2*Kd3*LsT + 3*Kd1**2*Kd3*RT + Kd1**2*LT*LsT - Kd1**2*LT*RT - Kd1**2*LsT**2 + Kd1**2*LsT*RT - 3*Kd1*Kd2*Kd3**2 + Kd1*Kd2*Kd3*LsT - Kd1*Kd2*Kd3*RT - 3*Kd1*Kd3**2*LT - 2*Kd1*Kd3**2*LsT + 3*Kd1*Kd3**2*RT - Kd1*Kd3*LT*LsT + Kd1*Kd3*LT*RT - 2*Kd1*Kd3*LsT**2 + 3*Kd1*Kd3*LsT*RT - Kd1*Kd3*RT**2 - Kd2*Kd3**2*LsT + Kd2*Kd3**2*RT)
        d = Kd1**3*(3*Kd1**2*Kd2*Kd3 - Kd1**2*Kd3**2 + 3*Kd1**2*Kd3*LT - 2*Kd1**2*Kd3*LsT + 3*Kd1**2*Kd3*RT + 2*Kd1**2*LT*LsT - 3*Kd1**2*LT*RT - Kd1**2*LsT**2 + 2*Kd1**2*LsT*RT - 3*Kd1*Kd2*Kd3**2 + 2*Kd1*Kd2*Kd3*LsT - 3*Kd1*Kd2*Kd3*RT - 3*Kd1*Kd3**2*LT - Kd1*Kd3**2*LsT + 3*Kd1*Kd3**2*RT - 2*Kd1*Kd3*LT*LsT + 3*Kd1*Kd3*LT*RT - 2*Kd1*Kd3*LsT**2 + 6*Kd1*Kd3*LsT*RT - 3*Kd1*Kd3*RT**2 - Kd1*LsT**3 + 2*Kd1*LsT**2*RT - Kd1*LsT*RT**2 - 2*Kd2*Kd3**2*LsT + 3*Kd2*Kd3**2*RT)
        e = Kd1**4*(Kd1**2*Kd2*Kd3 + Kd1**2*Kd3*LT + Kd1**2*Kd3*RT + Kd1**2*LT*LsT - 3*Kd1**2*LT*RT + Kd1**2*LsT*RT - Kd1*Kd2*Kd3**2 + Kd1*Kd2*Kd3*LsT - 3*Kd1*Kd2*Kd3*RT - Kd1*Kd3**2*LT + Kd1*Kd3**2*RT - Kd1*Kd3*LT*LsT + 3*Kd1*Kd3*LT*RT + 3*Kd1*Kd3*LsT*RT - 3*Kd1*Kd3*RT**2 + 2*Kd1*LsT**2*RT - 2*Kd1*LsT*RT**2 - Kd2*Kd3**2*LsT + 3*Kd2*Kd3**2*RT)
        f = Kd1**5*RT*(-Kd1**2*LT - Kd1*Kd2*Kd3 + Kd1*Kd3*LT - Kd1*Kd3*RT - Kd1*LsT*RT + Kd2*Kd3**2)

        p = [a, b, c, d, e, f] 
        roots = np.roots(p) 
        R = roots[0]
        R = max(roots)
        
        Fsb = R/(Kd1 + R)

        Fsb_array.append(Fsb)
    
    return Fsb_array
