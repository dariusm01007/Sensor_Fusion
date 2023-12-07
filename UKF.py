import numpy as np

class UKF:
    
    # Constructor
    def __init__(self, states, covariance, alpha, beta):
        self.states = states
        self.covariance = covariance
        self.alpha = alpha
        self.beta = beta
        
    def sigmaPoints(self):
        
        x = self.states
        P = self.covariance
        alpha = self.alpha
    
        # Van der Merwe scaled sigma point implementation
        n = len(x)
        
        kappa = 3 - n
        
        lambda_ = (alpha**2 * (n + kappa)) - n
        
        # ========= Sigma Points =========
        
        sigmas = np.zeros((2*n + 1, n))
        
        # Matrix square root of P scaled by (n+λ)
        U = np.linalg.cholesky((n + lambda_)*P)
        
        # First row is the mean of (x)
        sigmas[0] = np.transpose(x)
        
        # Transposing to make the loop easier
        X = np.transpose(x)
        
        for k in range(n):
            
            sigmas[k+1] = X + U[k]
            
            sigmas[n+k+1] = X - U[k]
            
        # Flipping it so that the sigma points are column vectors    
        sigmas = np.transpose(sigmas)
        
        return sigmas
    
    def ukfCovariance(self, sigmas,muX,Wc,Q):
    
        n = (np.shape(sigmas)[0])  # rows
        
        n2 = (np.shape(sigmas)[1])   # cols
        
        P = np.zeros((n,n))
        
        for i in range(n2):
            
            y = sigmas[ :,i].reshape(n,1)  - muX
            
            P += Wc[0,i]* (y@np.transpose(y))
            
        return P + Q
    
    def crossCovariance(self, sigmasX,sigmasZ,muX,muZ, Wc):
        
        n = (np.shape(sigmasX)[0]) # rows (states)
        
        m = (np.shape(sigmasZ)[0]) # rows (sensors)
        
        Pxz = np.zeros((n,m)) # Pxz = (states x sensors)
        
        n2 = (np.shape(sigmasX)[1]) # cols (sensors and states have the same number)
        
        for i in range(n2):
            
            x = sigmasX[ :,i].reshape(n,1)  - muX
            
            z = sigmasZ[ :,i].reshape(m,1)  - muZ
            
            Pxz += Wc[0,i]* (x@np.transpose(z))
        
        return Pxz
    
    def weights(self):
        
        x = self.states
        alpha = self.alpha
        beta  = self.beta
    
        n = len(x)
        
        kappa = 3-n
        
        # Weights
        
        lambda_ = (alpha**2 * (n + kappa)) - n
        
        # Covariance Weights
        Wc = np.zeros((1, 2*n + 1))
        
        Wc = np.full(2*n+1, 1 / (2* (n + lambda_)))
        
        Wc[0] = lambda_ / (n+lambda_) + (1-alpha**2 + beta)
        
        # Mean (state) Weights
        Wm = np.zeros((1, 2*n + 1))
        
        Wm = np.full(2*n+1, 1 / (2* (n + lambda_)))
        
        Wm[0] = lambda_ / (n+lambda_)
        
        # Making sure they are the correct shape
        Wc = Wc.reshape((1, 2*n + 1))
        
        Wm = Wm.reshape((1, 2*n + 1))
        
        return Wm,Wc
    
    def linearStatePropagationUKF(self,sigmas,Wm,F,G,u):
    
        muX = np.zeros((F.shape[0],1)) # empty vector for now
        
        newSigmas = np.zeros(np.shape(sigmas))
    
        n = (np.shape(sigmas)[1])   # cols
        n2 = (np.shape(sigmas)[0])  # rows
        
        Gk = G@u
        
        for i in range(n):
            
            y = sigmas[:,i].reshape(n2,1)
            
            newSigmas[:,i] = (F@y.reshape(n2,)) + Gk.reshape(n2,)
            
            muX += Wm[0,i] * newSigmas[:,i].reshape(n2,1)
            
        return muX
    
    def CovarianceUpdate(self, Px, Pz, K):
            
        #a = np.dot(K,Pz)
        #b = np.dot(a, np.transpose(K))
        
        b = (K@Pz)@np.transpose(K)
    
        return Px - b
    
            
            