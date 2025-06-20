### HEADER
import numpy as np
import kinematics as kin

### Constants ### 

mB0, mK0, mD0, mPi0 = np.array([5279.65, 497.611, 1864.84, 134.976])* 10**(-3) # Meson masses in GeV;

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
            p3Lab = kin.p3_lab(m1,m2,m3,theta1_lab,p1_lab,theta,phi)
            result[itheta,iphi,0] = np.log10(kin.getTheta(p3Lab))
            result[itheta,iphi,1] = np.log10(kin.getMom(p3Lab))
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