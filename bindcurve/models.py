import numpy as np


def get_list_of_models(kind):
    
    if kind == "logistic":
        models = ["IC50", "logIC50"]
        
    if kind == "Kd_competition":
        models = ["comp_3st_specific", "comp_3st_total", "comp_4st_specific", "comp_4st_total"]
        
    if kind == "Kd_direct":
        models = ["dir_simple", "dir_specific", "dir_total"]
                
    if kind == "conversion":
        models = ["cheng_prusoff","cheng_prusoff_corr","coleska"]
    
    return models




def IC50(LT, ymin, ymax, slope, IC50):

    model = ymin + (ymax - ymin)/(1 + (IC50/LT)**slope)
    
    return model

def IC50_lmfit(params, x, data=None):
    ymin = params['ymin']
    ymax = params['ymax']
    IC50_par = params['IC50']
    slope = params['slope']
    
    model = IC50(x, ymin, ymax, slope, IC50_par)
    
    if data is None:
        return model
    else:
        return model-data
    
    
    
def logIC50(LT, ymin, ymax, slope, logIC50):
    
    model = ymin + (ymax - ymin)/(1 + 10**((logIC50-LT)*slope))
    
    return model


def logIC50_lmfit(params, x, data=None):
    ymin = params['ymin']
    ymax = params['ymax']
    logIC50_par = params['logIC50']
    slope = params['slope']
    
    model = logIC50(x, ymin, ymax, slope, logIC50_par)
    
    if data is None:
        return model
    else:
        return model-data



def dir_simple(RT, ymin, ymax, Kds):
    
    R = RT
    Fsb = R/(Kds+R)
    model = ymin + (ymax - ymin)*Fsb
    return model

def dir_simple_lmfit(params, x, data=None):
    ymin = params['ymin']
    ymax = params['ymax']
    Kds = params['Kds']
    
    model = dir_simple(x, ymin, ymax, Kds)
    
    if data is None:
        return model
    else:
        return model-data



def dir_specific(RT, ymin, ymax, LsT, Kds):
    
    a = Kds + LsT - RT
    b = -Kds*RT
    
    R = (-a + np.sqrt(a**2 - 4*b))/2

    Fsb = R/(Kds+R)
    model = ymin + (ymax - ymin)*Fsb
    
    # other components
    RLs = RT - R
    Ls = LsT - RLs

    return model, R, RLs, Ls


def dir_specific_lmfit(params, x, data=None):
    ymin = params['ymin']
    ymax = params['ymax']
    LsT = params['LsT']
    Kds = params['Kds']

    model = dir_specific(x, ymin, ymax, LsT, Kds)[0]
    
    if data is None:
        return model
    else:
        return model-data



def dir_total(RT, ymin, ymax, LsT, Kds, Ns):
    
    a = (1+Ns)*Kds + LsT - RT
    b = -Kds*RT*(1+Ns)
    
    R = (-a + np.sqrt(a**2 - 4*b))/2

    Fsb = R/(Kds+R)
    model = ymin + (ymax - ymin)*Fsb
    
    # other components
    RLs = RT - R
    Ls = LsT - RLs

    return model, R, RLs, Ls




def dir_total_lmfit(params, x, data=None):
    ymin = params['ymin']
    ymax = params['ymax']
    LsT = params['LsT']
    Kds = params['Kds']
    Ns = params['Ns']

    model = dir_total(x, ymin, ymax, LsT, Kds, Ns)[0]
    
    if data is None:
        return model
    else:
        return model-data





# An exact mathematical expression for describing competitive binding of two different ligands to a protein molecule
# https://doi.org/10.1016/0014-5793(95)00062-E
def comp_3st_specific(LT, ymin, ymax, RT, LsT, Kds, Kd): 

    a = Kds + Kd + LsT + LT - RT
    b = Kds*(LT - RT) + Kd*(LsT - RT) + Kds*Kd
    c = -Kds*Kd*RT
    
    theta = np.acos((-2*a**3 + 9*a*b - 27*c)/(2*np.sqrt((a**2 - 3*b)**3)))
   
    R = -(a/3) + (2/3)*np.sqrt(a**2 - 3*b)*np.cos(theta/3)

    Fsb = R/(Kds + R)
    model = ymin + (ymax - ymin)*Fsb 
       
    # Other components
    RLs = (LsT*(2*np.sqrt(a**2 - 3*b)*np.cos(theta/3)-a)) / (3*Kds + (2*np.sqrt(a**2 - 3*b)*np.cos(theta/3)-a))
    RL = (LT*(2*np.sqrt(a**2 - 3*b)*np.cos(theta/3)-a)) / (3*Kd + (2*np.sqrt(a**2 - 3*b)*np.cos(theta/3)-a))
    Ls = LsT - RLs
    L = LT - RL
    
    return model, R, RLs, RL, Ls, L


def comp_3st_specific_lmfit(params, x, data=None): 
    ymin = params['ymin']
    ymax = params['ymax']
    RT = params['RT']
    LsT = params['LsT']
    Kds = params['Kds']
    Kd = params['Kd']
    
    model = comp_3st_specific(x, ymin, ymax, RT, LsT, Kds, Kd)[0]
    
    if data is None:
        return model
    else:
        return model-data


def comp_3st_total(LT, ymin, ymax, RT, LsT, Kds, Kd, N): 

    a = Kds + (1+N)*Kd + LsT + LT - RT
    b = Kds*(LT - RT) + (1+N)*Kd*(LsT - RT) + (1+N)*Kds*Kd
    c = -Kds*Kd*RT*(1+N)
    
    theta = np.acos((-2*a**3 + 9*a*b - 27*c)/(2*np.sqrt((a**2 - 3*b)**3)))
   
    R = -(a/3) + (2/3)*np.sqrt(a**2 - 3*b)*np.cos(theta/3)
    
    Fsb = R/(Kds + R)
    model = ymin + (ymax - ymin)*Fsb
    
    # Other components
    RLs = (LsT*(2*np.sqrt(a**2 - 3*b)*np.cos(theta/3)-a)) / (3*Kds + (2*np.sqrt(a**2 - 3*b)*np.cos(theta/3)-a))
    RL = (LT*(2*np.sqrt(a**2 - 3*b)*np.cos(theta/3)-a)) / (3*(1+N)*Kd + (2*np.sqrt(a**2 - 3*b)*np.cos(theta/3)-a))
    Ls = LsT - RLs
    L = LT - RL
    
    return model, R, RLs, RL, Ls, L


def comp_3st_total_lmfit(params, x, data=None): 
    ymin = params['ymin']
    ymax = params['ymax']
    RT = params['RT']
    LsT = params['LsT']
    Kds = params['Kds']
    Kd = params['Kd']
    N = params['N']
    
    model = comp_3st_total(x, ymin, ymax, RT, LsT, Kds, Kd, N)[0]

    if data is None:
        return model
    else:
        return model-data



def comp_4st_specific(LT, ymin, ymax, RT, LsT, Kds, Kd, Kd3): 
    
    Fsb_array = []
    
    for LT in LT:
        a = -Kds**2*Kd3**2
        b = Kds**2*Kd3*(Kds*Kd - 3*Kds*Kd3 + Kds*LT - 2*Kds*LsT + Kds*RT - Kd*Kd3 - Kd3*LT - Kd3*LsT + Kd3*RT)
        c = Kds**2*(3*Kds**2*Kd*Kd3 - 3*Kds**2*Kd3**2 + 3*Kds**2*Kd3*LT - 4*Kds**2*Kd3*LsT + 3*Kds**2*Kd3*RT + Kds**2*LT*LsT - Kds**2*LT*RT - Kds**2*LsT**2 + Kds**2*LsT*RT - 3*Kds*Kd*Kd3**2 + Kds*Kd*Kd3*LsT - Kds*Kd*Kd3*RT - 3*Kds*Kd3**2*LT - 2*Kds*Kd3**2*LsT + 3*Kds*Kd3**2*RT - Kds*Kd3*LT*LsT + Kds*Kd3*LT*RT - 2*Kds*Kd3*LsT**2 + 3*Kds*Kd3*LsT*RT - Kds*Kd3*RT**2 - Kd*Kd3**2*LsT + Kd*Kd3**2*RT)
        d = Kds**3*(3*Kds**2*Kd*Kd3 - Kds**2*Kd3**2 + 3*Kds**2*Kd3*LT - 2*Kds**2*Kd3*LsT + 3*Kds**2*Kd3*RT + 2*Kds**2*LT*LsT - 3*Kds**2*LT*RT - Kds**2*LsT**2 + 2*Kds**2*LsT*RT - 3*Kds*Kd*Kd3**2 + 2*Kds*Kd*Kd3*LsT - 3*Kds*Kd*Kd3*RT - 3*Kds*Kd3**2*LT - Kds*Kd3**2*LsT + 3*Kds*Kd3**2*RT - 2*Kds*Kd3*LT*LsT + 3*Kds*Kd3*LT*RT - 2*Kds*Kd3*LsT**2 + 6*Kds*Kd3*LsT*RT - 3*Kds*Kd3*RT**2 - Kds*LsT**3 + 2*Kds*LsT**2*RT - Kds*LsT*RT**2 - 2*Kd*Kd3**2*LsT + 3*Kd*Kd3**2*RT)
        e = Kds**4*(Kds**2*Kd*Kd3 + Kds**2*Kd3*LT + Kds**2*Kd3*RT + Kds**2*LT*LsT - 3*Kds**2*LT*RT + Kds**2*LsT*RT - Kds*Kd*Kd3**2 + Kds*Kd*Kd3*LsT - 3*Kds*Kd*Kd3*RT - Kds*Kd3**2*LT + Kds*Kd3**2*RT - Kds*Kd3*LT*LsT + 3*Kds*Kd3*LT*RT + 3*Kds*Kd3*LsT*RT - 3*Kds*Kd3*RT**2 + 2*Kds*LsT**2*RT - 2*Kds*LsT*RT**2 - Kd*Kd3**2*LsT + 3*Kd*Kd3**2*RT)
        f = Kds**5*RT*(-Kds**2*LT - Kds*Kd*Kd3 + Kds*Kd3*LT - Kds*Kd3*RT - Kds*LsT*RT + Kd*Kd3**2)

        p = [a, b, c, d, e, f] 
        roots = np.roots(p) 
        R = np.max(roots)   # takes the max root
        R = np.real(R)      # returns the real part if the root of R is complex
        
        Fsb = R/(Kds + R)

        Fsb_array.append(Fsb)
        
    Fsb_array = np.array(Fsb_array)    
    model = ymin + (ymax - ymin)*Fsb_array
    
    return model


def comp_4st_specific_lmfit(params, x, data=None): 
    ymin = params['ymin']
    ymax = params['ymax']
    RT = params['RT']
    LsT = params['LsT']
    Kds = params['Kds']
    Kd = params['Kd']
    Kd3 = params['Kd3']
      
    model = comp_4st_specific(x, ymin, ymax, RT, LsT, Kds, Kd, Kd3)
    
    if data is None:
        return model
    else:
        return model-data




def comp_4st_total(LT, ymin, ymax, RT, LsT, Kds, Kd, Kd3, N): 
    
    Fsb_array = []
    for LT in LT:
        a = -Kds**2*Kd3**2
        b = Kds**2*Kd3*((1+N)*Kds*Kd - 3*Kds*Kd3 + Kds*LT - 2*Kds*LsT + Kds*RT - (1+N)*Kd*Kd3 - Kd3*LT - Kd3*LsT + Kd3*RT)
        c = Kds**2*((1+N)*3*Kds**2*Kd*Kd3 - 3*Kds**2*Kd3**2 + 3*Kds**2*Kd3*LT - 4*Kds**2*Kd3*LsT + 3*Kds**2*Kd3*RT + Kds**2*LT*LsT - Kds**2*LT*RT - Kds**2*LsT**2 + Kds**2*LsT*RT - (1+N)*3*Kds*Kd*Kd3**2 + (1+N)*Kds*Kd*Kd3*LsT - (1+N)*Kds*Kd*Kd3*RT - 3*Kds*Kd3**2*LT - 2*Kds*Kd3**2*LsT + 3*Kds*Kd3**2*RT - Kds*Kd3*LT*LsT + Kds*Kd3*LT*RT - 2*Kds*Kd3*LsT**2 + 3*Kds*Kd3*LsT*RT - Kds*Kd3*RT**2 - (1+N)*Kd*Kd3**2*LsT + (1+N)*Kd*Kd3**2*RT)
        d = Kds**3*((1+N)*3*Kds**2*(1+N)*Kd*Kd3 - Kds**2*Kd3**2 + 3*Kds**2*Kd3*LT - 2*Kds**2*Kd3*LsT + 3*Kds**2*Kd3*RT + 2*Kds**2*LT*LsT - 3*Kds**2*LT*RT - Kds**2*LsT**2 + 2*Kds**2*LsT*RT - (1+N)*3*Kds*Kd*Kd3**2 + (1+N)*2*Kds*Kd*Kd3*LsT - (1+N)*3*Kds*Kd*Kd3*RT - 3*Kds*Kd3**2*LT - Kds*Kd3**2*LsT + 3*Kds*Kd3**2*RT - 2*Kds*Kd3*LT*LsT + 3*Kds*Kd3*LT*RT - 2*Kds*Kd3*LsT**2 + 6*Kds*Kd3*LsT*RT - 3*Kds*Kd3*RT**2 - Kds*LsT**3 + 2*Kds*LsT**2*RT - Kds*LsT*RT**2 - (1+N)*2*Kd*Kd3**2*LsT + (1+N)*3*Kd*Kd3**2*RT)
        e = Kds**4*((1+N)*Kds**2*Kd*Kd3 + Kds**2*Kd3*LT + Kds**2*Kd3*RT + Kds**2*LT*LsT - 3*Kds**2*LT*RT + Kds**2*LsT*RT - (1+N)*Kds*Kd*Kd3**2 + (1+N)*Kds*Kd*Kd3*LsT - (1+N)*3*Kds*Kd*Kd3*RT - Kds*Kd3**2*LT + Kds*Kd3**2*RT - Kds*Kd3*LT*LsT + 3*Kds*Kd3*LT*RT + 3*Kds*Kd3*LsT*RT - 3*Kds*Kd3*RT**2 + 2*Kds*LsT**2*RT - 2*Kds*LsT*RT**2 - (1+N)*Kd*Kd3**2*LsT + (1+N)*3*Kd*Kd3**2*RT)
        f = Kds**5*RT*(-Kds**2*LT - (1+N)*Kds*Kd*Kd3 + Kds*Kd3*LT - Kds*Kd3*RT - Kds*LsT*RT + (1+N)*Kd*Kd3**2)

        p = [a, b, c, d, e, f] 
        roots = np.roots(p) 
        R = np.max(roots)   # takes the max root
        R = np.real(R)      # returns the real part if the root of R is complex
        
        Fsb = R/(Kds + R)

        Fsb_array.append(Fsb)
        
    Fsb_array = np.array(Fsb_array)    
    model = ymin + (ymax - ymin)*Fsb_array
    return model



def comp_4st_total_lmfit(params, x, data=None): 
    ymin = params['ymin']
    ymax = params['ymax']
    RT = params['RT']
    LsT = params['LsT']
    Kds = params['Kds']
    Kd = params['Kd']
    Kd3 = params['Kd3']
    N = params['N']
    
    model = comp_4st_total(x, ymin, ymax, RT, LsT, Kds, Kd, Kd3, N)
    
    if data is None:
        return model
    else:
        return model-data












def cheng_prusoff(LsT, Kds, IC50):
    
    Kd = IC50/(1 + (LsT/Kds))
    
    return Kd


# Reference: https://doi.org/10.3109/10799898809049010
def cheng_prusoff_corr(LsT, Kds, y0, IC50):
    
    Kd = IC50/(1 + (LsT*(y0+2)/2*Kds*(y0+1) + y0)) + Kds*(y0/(y0+2))
    
    return Kd


# Reference: https://doi.org/10.1016/j.ab.2004.05.055
def coleska(RT, LsT, Kds, IC50):
    
    a = LsT + Kds - RT
    b = -Kds*RT
    
    # solving for R0 by quadratic formula
    R0 = (-a + np.sqrt(a**2 - 4*b)) / 2

    # Calculating the rest of the terms
    Ls0 = LsT/(1+(R0/Kds))
    RLs0 = RT/(1+(Kds/Ls0))
    RLs50 = RLs0 / 2
    Ls50 = LsT - RLs50
    RL50 = RT + (Kds*(RLs50/Ls50)) + RLs50
    L50 = IC50 - RL50
    Kd = L50/((Ls50/Kds) + (R0/Kds) + 1)
    
    return Kd