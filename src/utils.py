"""
utils.py
This script contains functions for generating diffusion simulations, 
data generators needed for the network training/testing, and other necessary 
functions.
"""


import numpy as np
from scipy import stats,fftpack,interpolate
from keras.utils import to_categorical
from stochastic.processes import diffusion
from stochastic.processes.diffusion import OrnsteinUhlenbeckProcess
import scipy.io
import matplotlib.pyplot as plt



"""
Function autocorr calculates the autocorrelation of a given input vector x

Input: 
    x - 1D vector 
    
Outputs:
    autocorr(x)    
"""

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[np.int(result.size/2):]


"""
Function OrnsteinUng generates a single realization of the Ornstein–Uhlenbeck 
noise process
see https://stochastic.readthedocs.io/en/latest/diffusion.html#stochastic.diffusion.OrnsteinUhlenbeckProcess
for more details.

Input: 
    n - number of points to generate
    T - End time
    speed - speed of reversion 
    mean - mean of the process
    vol - volatility coefficient of the process
    
Outputs:
    x - Ornstein Uhlenbeck process realization
"""

def OrnsteinUng(n=1000,T=50,speed=0,mean=0,vol=0):
    OU = OrnsteinUhlenbeckProcess(speed=speed,vol=vol,t=T)
    x = OU.sample(n=n)
    
    return x

#%% 
'''
function fbm_diffusion generates FBM diffusion trajectory (x,y,t)
realization is based on the Circulant Embedding method presented in:
Schmidt, V., 2014. Stochastic geometry, spatial statistics and random fields. Springer.

Input: 
    n - number of points to generate
    H - Hurst exponent
    T - end time
    
Outputs:
    x - x axis coordinates
    y - y axis coordinates
    t - time points
        
'''
def fbm_diffusion(n=1000,H=1,T=15):

    # first row of circulant matrix
    r = np.zeros(n+1)
    r[0] = 1
    idx = np.arange(1,n+1,1)
    r[idx] = 0.5*((idx+1)**(2*H) - 2*idx**(2*H) + (idx-1)**(2*H))
    r = np.concatenate((r,r[np.arange(len(r)-2,0,-1)]))
    
    # get eigenvalues through fourier transform
    lamda = np.real(fftpack.fft(r))/(2*n)
    
    # get trajectory using fft: dimensions assumed uncoupled
    x = fftpack.fft(np.sqrt(lamda)*(np.random.normal(size=(2*n)) + 1j*np.random.normal(size=(2*n))))
    x = n**(-H)*np.cumsum(np.real(x[:n])) # rescale
    x = ((T**H)*x)# resulting traj. in x
    y = fftpack.fft(np.sqrt(lamda)*(np.random.normal(size=(2*n)) + 1j*np.random.normal(size=(2*n))))
    y = n**(-H)*np.cumsum(np.real(y[:n])) # rescale
    y = ((T**H)*y) # resulting traj. in y

    t = np.arange(0,n+1,1)/n
    t = t*T # scale for final time T
    

    return x,y,t

'''
CTRW diffusion - generate CTRW trajectory (x,y,t)
function based on mittag-leffler distribution for waiting times and 
alpha-levy distribution for spatial lengths.
for more information see: 
Fulger, D., Scalas, E. and Germano, G., 2008. 
Monte Carlo simulation of uncoupled continuous-time random walks yielding a 
stochastic solution of the space-time fractional diffusion equation. 
Physical Review E, 77(2), p.021122.

Inputs: 
    n - number of points to generate
    alpha - exponent of the waiting time distribution function 
    gamma  - scale parameter for the mittag-leffler and alpha stable distributions.
    T - End time
'''
# Generate mittag-leffler random numbers
def mittag_leffler_rand(beta=0.5, n=1000, gamma=1):
    t = -np.log(np.random.uniform(size=[n, 1]))
    u = np.random.uniform(size=[n, 1])
    w = np.sin(beta * np.pi) / np.tan(beta * np.pi * u) - np.cos(beta * np.pi)
    t = t * w**(1. / beta)
    t = gamma * t

    return t


# Generate symmetric alpha-levi random numbers
def symmetric_alpha_levy(alpha=0.5, n=1000, gamma=1):
    u = np.random.uniform(size=[n, 1])
    v = np.random.uniform(size=[n, 1])

    phi = np.pi * (v - 0.5)
    w = np.sin(alpha * phi) / np.cos(phi)
    z = -1 * np.log(u) * np.cos(phi)
    z = z / np.cos((1 - alpha) * phi)
    x = gamma * w * z**(1 - (1 / alpha))

    return x


# needed for CTRW
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# Generate CTRW diffusion trajectory
def CTRW(n=1000, alpha=1, gamma=1, T=40):
    '''
    CTRW diffusion - generate CTRW trajectory (x,y,t)
    function based on mittag-leffler distribution for waiting times and
    alpha-levy distribution for spatial lengths.
    for more information see:
    Fulger, D., Scalas, E. and Germano, G., 2008.
    Monte Carlo simulation of uncoupled continuous-time random walks yielding a
    stochastic solution of the space-time fractional diffusion equation.
    Physical Review E, 77(2), p.021122.

    https://en.wikipedia.org/wiki/Lévy_distribution
    https://en.wikipedia.org/wiki/Mittag-Leffler_distribution

    Inputs:
        n - number of points to generate
        alpha - exponent of the waiting time distribution function
        gamma  - scale parameter for the mittag-leffler and alpha stable
                 distributions.
        T - End time
    '''
    jumpsX = mittag_leffler_rand(alpha, n, gamma)

    rawTimeX = np.cumsum(jumpsX)
    tX = rawTimeX * (T) / np.max(rawTimeX)
    tX = np.reshape(tX, [len(tX), 1])

    x = symmetric_alpha_levy(alpha=2, n=n, gamma=gamma**(alpha / 2))
    x = np.cumsum(x)
    x = np.reshape(x, [len(x), 1])

    y = symmetric_alpha_levy(alpha=2, n=n, gamma=gamma**(alpha / 2))
    y = np.cumsum(y)
    y = np.reshape(y, [len(y), 1])

    tOut = np.arange(0, n, 1) * T / n
    xOut = np.zeros([n, 1])
    yOut = np.zeros([n, 1])
    for i in range(n):
        xOut[i, 0] = x[find_nearest(tX, tOut[i]), 0]
        yOut[i, 0] = y[find_nearest(tX, tOut[i]), 0]
    return xOut.T[0], yOut.T[0], tOut


'''
Brownian - generate Brownian motion trajectory (x,y)

Inputs: 
    N - number of points to generate
    T - End time 
    delta - Diffusion coefficient

Outputs:
    out1 - x axis values for each point of the trajectory
    out2 - y axis values for each point of the trajectory
'''

def Sub_brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    # generate a sample of n numbers from a normal distribution.
    r = stats.norm.rvs(size=x0.shape + (n,), scale=delta*np.sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)
    # Compute Brownian motion by forming the cumulative sum of random samples. 
    np.cumsum(r, axis=-1, out=out)
    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

def Brownian(N=1000,T=50,delta=1):
    x = np.empty((2,N+1))
    x[:, 0] = 0.0
    
    Sub_brownian(x[:,0], N, T/N, delta, out=x[:,1:])
    
    out1 = x[0][:N]
    out2 = x[1][:N]
    
    return out1,out2


#%%
'''
Generator functions for neural network training per Keras specifications
input for all functions is as follows:
    
input: 
   - batch size
   - steps: total number of steps in trajectory (list) 
   - T: final time (list)
   - sigma: Standard deviation of localization noise (std of a fixed cell/bead)
'''

def generate(batchsize=32,steps=1000,T=15,sigma=0.1,dilation=1,interpolate=-1):
    ##dilation represents number of steps to skip between points
    ##interpolate represents number of points to linearly interpolate between points (-1 indicates none)
    steps *= dilation

    ##essentially what was happening here was that I was increasing the number of sampled steps over some constant amount of time
    ##found that if we increase the number of sampled steps, and don't use all of them, we get a decrease in accuracy for some reason
    ##kind of weird that this is the case but it is possible that the simulation outputs steps in a different way if we ask for too many of them

    ##problem is interpolation: how do we interpolate by time? 
    ##We generate the tracks using these functions and we have the tracks in order: we have to do interpolation between points manually otherwise 

    ##theoretically sampling 30 points from 300 sampled ones should have relatively same accuracy as just 30 points when we sampled 30 but somehow this is not the case
    ##why: ??

    ##If we increase T and sample the same number of data points do things get worse??
    ##expectation: yes definitely

    first = True
    while True:
        # randomly choose a set of trajectory-length and final-time. This is intended
        # to increase variability in simuation conditions.


        # T1 = np.random.choice(T,size=1).item()
        T1 = T
        out = np.zeros([batchsize,steps//dilation-1,1])
        label = np.zeros([batchsize,1])
        for i in range(batchsize):
            # randomly select diffusion model to simulate for this iteration
            label[i,0] = np.random.choice([0,1,2])
            if label[i,0] == 0: 
                H = np.random.uniform(low=0.1,high=0.45) #subdiffusive
#                 coinflip = np.random.choice([0,1])
#                 if coinflip == 0:
#                    H = np.random.uniform(low=0.1,high=0.4) #subdiffusive
#                 else:
#                    H = np.random.uniform(low=0.6,high=0.99) #superdiffusive
                x,y,t = fbm_diffusion(n=steps,H=H,T=T1)                  
            elif label[i,0] == 1:
                x,y = Brownian(N=steps,T=T1,delta=1) 
            else:
                alpha=np.random.uniform(low=0.1,high=0.90)
                x,y,t = CTRW(n=steps,alpha=alpha,T=T1)
#                x,y,t = CTRW(n=steps,alpha=0.1,T=T1)

            # print(label[i,0])
            
            if interpolate <= 0:
                x = x[0::dilation]
                y = y[0::dilation]
            if interpolate > 0:
                x = x[0::dilation]
                y = y[0::dilation]

                # if i == 0 and first:
                #     print(x, y)
                #     print(len(x))

                num_interp_points = 50

                x_interp_points = [steps / num_interp_points]
                ##can't really interpolate between different times - we only have access to x points at different times

                y_interp_points = []
                for j in range(len(x_interp_points)):
                    y_interp_points.append(y[j])
                # f = scipy.interpolate.interp1d(x, y, kind="nearest")

                # y_interp_points = np.array([f(point) for point in x_interp_points])
                # # plt.figure()
                # plt.plot(x, y, label="original")
                # plt.plot(x_interp_points, f(x_interp_points), label="interpolated")
                # print(x_interp_points)
                # print(y_interp_points)
                # plt.legend()
                # plt.show()
                x_all = np.empty((x.size + x_interp_points.size), dtype=x.dtype)
                x_all[0::(interpolate + 1)] = x
                for j in range(interpolate):
                    x_all[(j + 1)::(interpolate + 1)] = x_interp_points[j::interpolate]
                x = x_all

                y_all = np.empty((y.size + y_interp_points.size), dtype=y.dtype)
                y_all[0::(interpolate + 1)] = y
                for j in range(interpolate):
                    y_all[(j + 1)::(interpolate + 1)] = y_interp_points[j::interpolate]
                y= y_all

                # if i == 0 and first:
                #     first = False
                #     print(label[i,0])
                #     print(x, y)
                #     print(len(x))
                # plt.figure()
                # plt.plot(x, y, label="interpolated")
                # plt.show()
                # print(label[i,0])
                # print(len(x))                
            # if interpolate <= 0:
            #     x = x[0::dilation]
            #     y = y[0::dilation]
            # if interpolate > 0:
            #     x = x[0::dilation]
            #     y = y[0::dilation]

            #     # if i == 0 and first:
            #     #     print(x, y)
            #     #     print(len(x))

            #     x_interp_points = (x[1:] + x[:-1]) / 2
            #     y_interp_points = []
            #     for j in range(len(x_interp_points)):
            #         y_interp_points.append(y[j])
            #     # f = scipy.interpolate.interp1d(x, y, kind="nearest")

            #     # y_interp_points = np.array([f(point) for point in x_interp_points])
            #     # # plt.figure()
            #     # plt.plot(x, y, label="original")
            #     # plt.plot(x_interp_points, f(x_interp_points), label="interpolated")
            #     # print(x_interp_points)
            #     # print(y_interp_points)
            #     # plt.legend()
            #     # plt.show()
            #     x_all = np.empty((x.size + x_interp_points.size), dtype=x.dtype)
            #     x_all[0::(interpolate + 1)] = x
            #     for j in range(interpolate):
            #         x_all[(j + 1)::(interpolate + 1)] = x_interp_points[j::interpolate]
            #     x = x_all

            #     y_all = np.empty((y.size + y_interp_points.size), dtype=y.dtype)
            #     y_all[0::(interpolate + 1)] = y
            #     for j in range(interpolate):
            #         y_all[(j + 1)::(interpolate + 1)] = y_interp_points[j::interpolate]
            #     y= y_all

            #     # if i == 0 and first:
            #     #     first = False
            #     #     print(label[i,0])
            #     #     print(x, y)
            #     #     print(len(x))
            #     # plt.figure()
            #     # plt.plot(x, y, label="interpolated")
            #     # plt.show()
            #     # print(label[i,0])
            #     # print(len(x))



            noise = np.sqrt(sigma)*np.random.randn(steps // dilation -1)
            x1 = np.reshape(x,[1,len(x)])
            x1 = x1-np.mean(x1)
            x_n = x1[0,:steps // dilation]
#            x_n = x1[0,:steps]
            dx = np.diff(x_n)
            # Generate OU noise to add to the data
#             nx = OrnsteinUng(n=steps-2,T=T1,speed=1,mean=0,vol=1)
#             dx = dx+sigma*nx
            if np.std(x) < 0.000001:
                dx = dx
            else:
                dx = dx/np.std(dx)
#            print(np.std(dx))
#            print(label[i,0])
                # print(dx.shape)
                # print(noise.shape)
                dx = dx+noise
                out[i,:,0] = dx
       
        label = to_categorical(label,num_classes=3)
#        return out,label
        yield out,label
        

'''
Generator functions for neural network training per Keras specifications
input for all functions is as follows:
    
input: 
   - batch size
   - steps: total number of steps in trajectory (list) 
   - T: final time (list)
   - sigma: Standard deviation of localization noise (std of a fixed cell/bead)
'''

# Randomly generate trajectories of different diffusion models for training of the 
# classification network
    
def generate_sim(batchsize=32,steps=1000,T=15,sigma=0.1):
    while True:
        # randomly choose a set of trajectory-length and final-time. This is intended
        # to increase variability in simuation conditions.
        T1 = np.random.choice(T,size=1).item()
        out = np.zeros([batchsize,steps-1,1])
        label = np.zeros([batchsize,1])
        for i in range(batchsize):
            # randomly select diffusion model to simulate for this iteration
            label[i,0] = np.random.choice([0,1,2])
            if label[i,0] == 0: 
                H = np.random.uniform(low=0.09,high=0.48) #subdiffusive
#                coinflip = np.random.choice([0,1])
#                if coinflip == 0:
#                   H = np.random.uniform(low=0.1,high=0.4) #subdiffusive
#                    H = 0.45
#                 else:
#                    H = np.random.uniform(low=0.6,high=0.99) #superdiffusive
                x,y,t = fbm_diffusion(n=steps,H=H,T=T1)
            elif label[i,0] == 1:
                 continue  
#                 x,y = Brownian(N=steps,T=T1,delta=1) 
            else:
                x,y,t = CTRW(n=steps,alpha=np.random.uniform(low=0.2,high=0.99),T=T1)
            noise = np.sqrt(sigma)*np.random.randn(1,steps)
            x1 = np.reshape(x,[1,len(x)])
            x1 = x1-np.mean(x1)     
#             x_n = x1[0,:steps]+noise
            x_n = x1[0,:steps]
            dx = np.diff(x_n)
            # Generate OU noise to add to the data
            nx = OrnsteinUng(n=steps-2,T=T1,speed=1,mean=0,vol=1)
            dx = dx+sigma*nx
            out[i,:,0] = dx
       
        return out,label
    
        
# generate FBM trajectories with different Hurst exponent values 
# for training of the Hurst-regression network
        
def fbm_regression(batchsize=32,steps=1000,T=[1],sigma=0.1):
    while True:
        # randomly choose a set of trajectory-length and final-time. This is intended
        # to increase variability in simuation conditions.
        T1 = np.random.choice(T,size=1).item()
        out = np.zeros([batchsize,steps-1,1])
        
        label = np.zeros([batchsize,1])
        for i in range(batchsize):
            H = np.random.uniform(low=0.01,high=0.48)
            label[i,0] = H
            x,y,t = fbm_diffusion(n=steps,H=H,T=T1)
            
            n = np.sqrt(sigma)*np.random.randn(steps,)
            x_n = x[:steps,]+n
            dx = np.diff(x_n,axis=0)
            
            out[i,:,0] = autocorr((dx-np.mean(dx))/(np.std(dx)))

        
        yield out,label
        
     
'''
Generate CTRW for CTRW single for finding alpha value
'''  
def generate_CTRW(batchsize=32,steps=1000,T=15,sigma=0.1):
    while True:
        # randomly choose a set of trajectory-length and final-time. This is intended
        # to increase variability in simuation conditions.
        T1 = np.random.choice(T,size=1).item()
        out = np.zeros([batchsize,steps-1,1])
        label = np.zeros([batchsize,1])
        for i in range(batchsize):
            alpha=np.random.uniform(low=0.1,high=0.99)
            label[i,0] = alpha
            x,y,t = CTRW(n=steps,alpha=alpha,T=T1)
            noise = np.sqrt(sigma)*np.random.randn(steps-1)
            x1 = np.reshape(x,[1,len(x)])
            x1 = x1-np.mean(x1)
            x_n = x1[0,:steps]
            dx = np.diff(x_n)
            if np.std(x) < 0.000001:
                dx = dx
            else:
                dx = dx/np.std(dx)
                dx = dx+noise
            out[i,:,0] = dx


        yield out, label       
