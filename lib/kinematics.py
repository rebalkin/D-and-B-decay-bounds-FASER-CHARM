### HEADER
import numpy as np

### Constants ### 

eta = np.diag([1,-1,-1,-1]);

mB0, mK0, mD0, mPi0 = np.array([5279.65, 497.611, 1864.84, 134.976])* 10**(-3) # Meson masses in GeV;

#################

### Functions related to kinematics ########################

def Lambda(m1, m2, m3):
    return (m1**2 - (m2 + m3)**2)*(m1**2 - (m2 - m3)**2)
  
### Momentum of (2) and (3) in the rest frame of (1) for the process 1 -> 2 3.
def Pcm(m1,m2,m3):
    return np.sqrt(Lambda(m1, m2, m3))/(2*m1);

### Energy of (2) in the rest frame of (1) for the process 1 -> 2 3. For the energy or particle 3, exchange m2 and m3.
def Ecm(m1,m2,m3):
    return (m1**2+m2**2-m3**2)/(2*m1);

### 4-Momentum of (1) in the rest frame of (1)
def p1(m1):
    return np.array([m1,0,0,0])

### 4-Momentum of (2) in the rest frame of (1), pointed in arbitrary direction theta and phi
def p2(m1,m2,m3,theta,phi):
    return np.array([Ecm(m1, m2, m3),0,0,0])- Pcm(m1,m2,m3)*np.array([0,np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])

### 4-Momentum of (3) in the rest frame of (1), pointed in arbitrary direction theta and phi
def p3(m1,m2,m3,theta,phi):
    return np.array([Ecm(m1, m3, m2),0,0,0])+ Pcm(m1,m2,m3)*np.array([0,np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])

### contruct a cartesian 4-Momentum from mass, momentum, theta and phi                                   
def pVec(m, p, theta, phi):
    return np.array([np.sqrt(p**2+m**2),0,0,0])+ p*np.array([0,np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
                                
### Boost a particle of mass m1 in the Z direction such that it has momentum p1_lab (defined as the lab frame)
def Oboost(p1_lab,m1):
    mat = [[np.sqrt(1+p1_lab**2/m1**2), 0, 0, p1_lab/m1],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [p1_lab/m1, 0, 0, np.sqrt(1+p1_lab**2/m1**2)]]
    return np.array(mat);

### rotate a particle moving in the Z direction to a direction defined by that and phi
def Orotate(theta,phi):
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    mat1 = [[1,0,0,0],
           [0,cphi,-sphi,0],
           [0,-sphi,cphi,0],
           [0,0,0,1]]
    mat2 = [[1,0,0,0],
           [0,ctheta,0,stheta],
           [0,0,1,0],
           [0,-stheta,0,ctheta]]
    return np.matmul(np.array(mat1),np.array(mat2))
    
### combine both transformation - transforms momenta from the rest frame of (1) to the lab frame where (1) has momentum p1_lab in a direction defined by theta and phi.
def Olab(p1_lab,m1,theta,phi):
    return np.matmul(Orotate(theta,phi),Oboost(p1_lab,m1))
  
### 4-Momentum of (1) in the lab frame
def p1_lab(m1,m2,m3,theta1_lab,p1_lab):
    return np.matmul( Olab(p1_lab,m1,theta1_lab,0),p1(m1))
  
### 4-Momentum of (2) in the lab frame
def p2_lab(m1,m2,m3,theta1_lab,p1_lab,theta,phi):
    return np.matmul( Olab(p1_lab,m1,theta1_lab,0),p2(m1,m2,m3,theta,phi))

### 4-Momentum of (3) in the lab frame
def p3_lab(m1,m2,m3,theta1_lab,p1_lab,theta,phi):
    return np.matmul( Olab(p1_lab,m1,theta1_lab,0),p3(m1,m2,m3,theta,phi))

def getMom(p):
    return np.sqrt((p[1])**2+(p[2])**2+(p[3])**2)

def getTheta(p):
    theta = np.arctan2(np.sqrt(p[1]**2+p[2]**2),p[3]) 
    return np.abs(theta)

    

### Generate weighted spectrum from a single meson spectrum bin 

def decay_spectrum_1_bin(m1,m2,m3,theta1_lab,p1_lab,binWeight,Ntheta,Nphi,eps):
    ## values for 4pi solid angle spread for decaying particle in the rest frame of (1)
    costheta_R = np.linspace(-1+eps,1-eps,Ntheta) #probing the edge values usually generates undesired artifacts (particle is produced colinear with parent), we avoid it by defining a threshold eps 
    phi_R = np.linspace(0,2*np.pi,Nphi)
    ## spread the total bin weight across that 4pi solid angle
    weight = binWeight/(Ntheta*Nphi)
    result = np.empty([int(Ntheta),int(Nphi),3],dtype=float)
    ## creating the spectrum of the decay particle {theta_x,p_x,weight} in the lab frame
    for itheta, costheta in enumerate(costheta_R):
        for iphi, phi in enumerate(phi_R):
            theta = np.arccos(costheta)
            p3Lab = p3_lab(m1,m2,m3,theta1_lab,p1_lab,theta,phi)
            result[itheta,iphi,0] = np.log10(getTheta(p3Lab))
            result[itheta,iphi,1] = np.log10(getMom(p3Lab))
            result[itheta,iphi,2] = weight
    return result.reshape(-1,3)

### B -> K (3) decays - also hard coding the spacing of theta and phi in the rest frame of the (B)

def spectrum_1binBtoKX(m3,theta1_lab,p1_lab,binWeight): 

    eps=10**(-6)
    Ntheta = 100
    Nphi = 50
    return decay_spectrum_1_bin(mB0,mK0,m3,theta1_lab,p1_lab,binWeight,Ntheta,Nphi,eps)

### D -> pi (3) decays - also hard coding the spacing of theta and phi in the rest frame of the (D)

def spectrum_1binDtoPiX(m3,theta1_lab,p1_lab,binWeight): 

    eps=10**(-6)
    Ntheta = 100
    Nphi = 50
    return decay_spectrum_1_bin(mD0,mPi0,m3,theta1_lab,p1_lab,binWeight,Ntheta,Nphi,eps)



