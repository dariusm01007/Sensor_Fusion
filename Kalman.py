import numpy as np
from numpy.linalg import inv
#import math


class Kalman:
    
    # Constructor
    def __init__(self, states, stateTransition, covariance, observation, processNoise, measurementNoise):
        self.states = states
        self.stateTransition = stateTransition
        self.covariance = covariance
        self.observation = observation
        self.processNoise = processNoise
        self.measurementNoise = measurementNoise
        
        
    def propagateStates(self, input):
        # xk(-) = F*xk-1(+) + Gk*uk
        self.states = self.stateTransition@self.states + input
    
    
    def propagateCovariance(self):
        
        F = self.stateTransition
        Pk = self.covariance
        Qk = self.processNoise
        
        a = F@Pk
        
        self.covariance = (a@np.transpose(F)) + Qk
    
    def innovationCovariance(self):
        
        H = self.observation
        Mk = self.covariance
        R = self.measurementNoise
        
        a = H@Mk
        
        return (a@np.transpose(H)) + R
    
    def covarianceUpdate(self, K):
        
        Pk = self.covariance
        H = self.observation
        R = self.measurementNoise
        
        # Joseph Form
        
        I = np.identity(Pk.shape[0])
        
        a = I - np.dot(K,H)
        b = np.dot(a,Pk)
        c = np.dot(b, np.transpose(a))

        d = np.dot(K, R)
        e = np.dot(d, np.transpose(K))     
        
        # (I - KH)Pkp(I - KH).' + (KRK.')
        
        self.covariance =  c + e
    
    def KalmanGain(self, S):
        
        Mk = self.covariance
        
        H = self.observation
        
        return (Mk@np.transpose(H))@inv(S)
    
    def stateUpdate(self, K, zk):
        
        H = self.observation
        residual = zk - H@self.states
        
        self.states = self.states + K@residual