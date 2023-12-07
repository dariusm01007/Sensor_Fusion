from Sensors import IMU, GPS, Truth
import numpy as np
#from UKF import UKF
from similarityTransform import getAngles, getPosition, getVelocity, getPosVel, getBias
from mathUtilities import Body2Inertial, IMU_AttitudeUpdate, gravityField, specificForceTransform, extrapolateMeasurement
from numpy.linalg import inv
import time
import matplotlib.pyplot as plt
import math
import os
import statistics as stats
from Kalman import Kalman
from UKF_Zarchan import UKF

# Adjustable settings
adpativeProcessNoise = True
adjustForBias = False
plotUncertainty = True

# Creating a folder to save the images and data from the simulation
home = os.getcwd()
savePath = home + "\simRuns2"

if not os.path.exists(savePath):
    print("\n{} does not currently exist... creating now:\n".format(savePath))
    os.makedirs(savePath)

fileName = savePath + "\stateStatistics.txt"
tmFileName = savePath + "\stateData.txt"

# Creating instances of classes related to this project
imu   = IMU()
gps   = GPS()
truth = Truth()

# Determining size for memory pre-allocation (for speed)
arrayLength = len(truth.time)

# Pre-Allocate Memory
posEast  = np.zeros(arrayLength)
posNorth = np.zeros(arrayLength)
posUp    = np.zeros(arrayLength)
velEast  = np.zeros(arrayLength)
velNorth = np.zeros(arrayLength)
velUp    = np.zeros(arrayLength)
yaw      = np.zeros(arrayLength)
pitch    = np.zeros(arrayLength)
roll     = np.zeros(arrayLength)
biasAx   = np.zeros(arrayLength)
biasAy   = np.zeros(arrayLength)
biasAz   = np.zeros(arrayLength)
biasGx   = np.zeros(arrayLength)
biasGy   = np.zeros(arrayLength)
biasGz   = np.zeros(arrayLength)

# State Residuals
x1Residual  = np.zeros(arrayLength)
x2Residual  = np.zeros(arrayLength)
x3Residual  = np.zeros(arrayLength)
x4Residual  = np.zeros(arrayLength)
x5Residual  = np.zeros(arrayLength)
x6Residual  = np.zeros(arrayLength)
x7Residual  = np.zeros(arrayLength)
x8Residual  = np.zeros(arrayLength)
x9Residual  = np.zeros(arrayLength)

# State Uncertainty
x1std  = np.zeros(arrayLength)
x2std  = np.zeros(arrayLength)
x3std  = np.zeros(arrayLength)
x4std  = np.zeros(arrayLength)
x5std  = np.zeros(arrayLength)
x6std  = np.zeros(arrayLength)
x7std  = np.zeros(arrayLength)
x8std  = np.zeros(arrayLength)
x9std  = np.zeros(arrayLength)

dt    = 0.01  # [s] -> IMU runs at 100 Hz
s     = 0     # [s] -> counter for when dt = 0.1 (GPS_AVAILABLE = True)
index = 0     # this is the last index (in the data) where the GPS data was valid

# Acceleration related variables
g = -9.81 # [m/s2]
inputAccel = np.zeros([3,1])
accelSum   = np.zeros([3,1])
avgAccel   = np.zeros([3,1])
gVector    = np.array([[0],
                       [0],
                       [g]])

# This will keep track of the number of GPS updates in between IMU measurements
count = 0 

# Flag representing we have GPS data 
GPS_AVAILABLE = False

"""  Initial conditions  """
x_I = gps.posEast[0]  # [m]
y_I = gps.posNorth[0] # [m]
z_I = gps.posUp[0]    # [m]

xdot_I = gps.velEast[0]  # [m/s]
ydot_I = gps.velNorth[0] # [m/s]
zdot_I = gps.velUp[0]    # [m/s]

phi    = np.deg2rad(truth.roll[0])  # [rad]
theta  = np.deg2rad(truth.pitch[0]) # [rad]
psi    = np.deg2rad(truth.yaw[0])   # [rad]

# Accelerometer Bias initialization
b_a_x = 0.001
b_a_y = 0.001
b_a_z = 0.001

# Gyroscope Bias initialization
b_g_x = 0.001
b_g_y = 0.001
b_g_z = 0.001

"""  Kalman Filter Setup """
# Number of standard deviations for residual bounds
numSigmas = 3

# 0 ≤ α ≤ 1
#alpha = 0.2
alpha = 0.5
# larger alpha scales the standard deviation and 
# spreads sigma points further from the mean 
#   (better for highly non-linear models)

beta = 2

# Initial states
xk_1 = np.array([[phi], 
                 [theta],
                 [psi],
                 [xdot_I],
                 [ydot_I],
                 [zdot_I],
                 [x_I],
                 [y_I],
                 [z_I],
                 [b_a_x],
                 [b_a_y],
                 [b_a_z],
                 [b_g_x],
                 [b_g_y],
                 [b_g_z]])

# Initial Covariance
P = np.identity(len(xk_1))
P[0][0] = np.deg2rad(1.0)  # Roll
P[1][1] = np.deg2rad(1.0)  # Pitch
P[2][2] = np.deg2rad(1.0)  # Yaw
P[3][3] = 20               # Velocity East
P[4][4] = 20               # Velocity North
P[5][5] = 20               # Velocity Up
P[6][6] = 50               # Position East
P[7][7] = 50               # Position North
P[8][8] = 100              # Position Up
Pk_1 = P@P

# Process Noise
Q = np.zeros([len(xk_1), len(xk_1)])
Q[0][0]   = 0.01 # Roll
Q[1][1]   = 0.01  # Pitch
Q[2][2]   = 0.01  # Yaw
Q[3][3]   = 0.01  # Velocity East
Q[4][4]   = 0.01  # Velocity North
Q[5][5]   = 3.0   # Velocity Up
Q[6][6]   = 0.01  # Position East
Q[7][7]   = 0.01  # Position North
Q[8][8]   = 0.01  # Position Up
Q[9][9]   = 0.0   # Accelerometer bias x
Q[10][10] = 0.0   # Accelerometer bias y
Q[11][11] = 0.0   # Accelerometer bias z
Q[12][12] = 0.0   # Gyroscope bias x
Q[13][13] = 0.0   # Gyroscope bias y
Q[14][14] = 0.0   # Gyroscope bias z
Qk = Q@Q
spectral = 0.100 # Scaling the process noise
Qk *= spectral

# Measurement Noise
R = np.identity(9)
R[0][0] = 0.01   # Roll
R[1][1] = 0.01   # Pitch
R[2][2] = 0.01   # Yaw
R[3][3] = 0.1   # Velocity East
R[4][4] = 0.1   # Velocity North
R[5][5] = 0.1   # Velocity Up
R[6][6] = 0.25  # Position East
R[7][7] = 0.25  # Position North
R[8][8] = 0.25  # Position Up
Rk = R@R

sensors = len(R)

# myUKF = UKF(xk_1,Pk_1,alpha,beta)
myUKF = UKF(xk_1,Pk_1)

# Biases
gyroBias  = np.zeros([3,1])
accelBias = np.zeros([3,1])

""" Accelerometer Kalman Filter """
# Initial states
x_a = np.array([[imu.accelX[0]],
              [0],
              [imu.accelY[0]],
              [0],
              [imu.accelZ[0]],
              [0]])

f_a = np.array([[1, dt],
                [0, 1]])

F_a = np.zeros([len(x_a), len(x_a)])
F_a[0:2,0:2] = f_a
F_a[2:4,2:4] = f_a
F_a[4:6,4:6] = f_a


P_a = np.array([[0.02, 0, 0,  0,  0,  0],
              [0, 0.02,  0,  0,  0,  0],
              [0, 0,  0.02,  0, 0,  0],
              [0, 0,  0,  0.02,  0,  0],
              [0, 0,  0,  0,  0.002,  0],
              [0, 0,  0,  0,  0,  0.002]])

H_a = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0]])

q_a = np.array([[ ((dt ** 3) / 3) , ((dt ** 2) / 2)],
              [((dt ** 2) / 2), dt] ]) 

Q_a = np.zeros([len(P_a), len(P_a)])

Q_a[0:2,0:2] = q_a
Q_a[2:4,2:4] = q_a
Q_a[4:6,4:6] = q_a

Qk_a = Q_a*0.0002

Rk_a = np.identity(3)
Rk_a[0,0] = 0.02
Rk_a[1,1] = 0.02
Rk_a[2,2] = 0.02

accelFilter = Kalman(x_a,F_a,P_a,H_a,Qk_a,Rk_a)
""" end accelerometer Kalman Filter """

""" Gyroscope Kalman Filter """
# Initial states
x_g = np.array([[imu.omegaX[0]],
              [0],
              [imu.omegaY[0]],
              [0],
              [imu.omegaZ[0]],
              [0]])

f_g = np.array([[1, dt],
                [0, 1]])

F_g = np.zeros([len(x_g), len(x_g)])
F_g[0:2,0:2] = f_g
F_g[2:4,2:4] = f_g
F_g[4:6,4:6] = f_g


P_g = np.array([[0.02, 0, 0,  0,  0,  0],
              [0, 0.02,  0,  0,  0,  0],
              [0, 0,  0.02,  0, 0,  0],
              [0, 0,  0,  0.02,  0,  0],
              [0, 0,  0,  0,  0.002,  0],
              [0, 0,  0,  0,  0,  0.002]])

H_g = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0]])

q_g = np.array([[ ((dt ** 3) / 3) , ((dt ** 2) / 2)],
              [((dt ** 2) / 2), dt] ]) 

Q_g = np.zeros([len(P_g), len(P_g)])

Q_g[0:2,0:2] = q_g
Q_g[2:4,2:4] = q_g
Q_g[4:6,4:6] = q_g

Qk_g = Q_g*0.0002

Rk_g = np.identity(3)
Rk_g[0,0] = 0.02
Rk_g[1,1] = 0.02
Rk_g[2,2] = 0.02

gyroFilter = Kalman(x_g,F_g,P_g,H_g,Qk_g,Rk_g)
""" end gyroscope Kalman Filter """

# Timing the algorithm (very slow)
print("Running algorithm...")
startTime = time.time()

# Creating a .txt file to store the state data 
with open(tmFileName, "w") as tmFile:
    tmFile.write("Time, Roll, Pitch, Yaw, Vx, Vy, Vz, Px, Py, Pz\n")
    
# Running the algorithm    
for idx in range (0,arrayLength):
    
    """ IMU Kalman Filters """
    #Predict States
    accelFilter.propagateStates(0.0)
    gyroFilter.propagateStates(0.0)
    
    # Propagate Uncertainty
    accelFilter.propagateCovariance()
    gyroFilter.propagateCovariance()
    
    # Innovation Covariance
    Sk_a = accelFilter.innovationCovariance()
    Sk_g = gyroFilter.innovationCovariance()
    
    # Kalman Gain
    K_accel = accelFilter.KalmanGain(Sk_a)
    K_gyro  = accelFilter.KalmanGain(Sk_g)
    
    # Measurement Data 
    zk_accel = np.array([[imu.accelX[idx]],
                          [imu.accelY[idx]],
                          [imu.accelZ[idx]]])
    
    zk_gyro = np.array([[imu.omegaX[idx]],
                          [imu.omegaY[idx]],
                          [imu.omegaZ[idx]]])
    
    # Update States
    accelFilter.stateUpdate(K_accel,zk_accel)
    gyroFilter.stateUpdate(K_gyro,zk_gyro)
    
    # Update Uncertainty
    accelFilter.covarianceUpdate(K_accel)
    gyroFilter.covarianceUpdate(K_gyro)
    """ end IMU  Kalman Filters """

    # Generating the sigma points based on the states and uncertainty
    sigmas = myUKF.sigmaPoints()

    # Weighing factors for the Unscented Transform
    Wm,Wc = myUKF.weights()
    
    n = (np.shape(sigmas)[1]) # cols (number of sigma points)
    m = (np.shape(sigmas)[0]) # rows (number of states)
    
    # Final prediction
    muX = np.zeros([m, 1])
    
    # Gathering the filtered Gyro measurments (Body Frame)
    BodyRates =  np.array([[gyroFilter.states[0].item()],
                           [gyroFilter.states[2].item()],
                           [gyroFilter.states[4].item()]])
    
    # Gathering the filtered Accelerometer measurments (Body Frame)
    specificForce_B = np.array([[accelFilter.states[0].item()],
                                [accelFilter.states[2].item()],
                                [accelFilter.states[4].item()]])
    
    # Using the estimated bias to make corrections
    BodyRates -= gyroBias
    
    specificForce_B -= accelBias
    
    # Incrementing the time until the next GPS update
    s += dt
    
    if s >= (10*dt - 0.00001):
        GPS_AVAILABLE = True
        # Reset the counter
        s = 0
    else:
        GPS_AVAILABLE = False
        
    
    for i in range (n):
        
        # Grabbing a set of sigma points to do the navigation calculations
        states = sigmas[:,i].reshape(m,1)
        
        """ 1. Attitude Update  """
        currentAngles = getAngles(states)
        phi   = currentAngles[0].item()
        theta = currentAngles[1].item()
        psi   = currentAngles[2].item()
        
        theDCM = Body2Inertial (phi, theta, psi)
        
        IMU_angles = IMU_AttitudeUpdate(theDCM, BodyRates, dt) # output is in radians 
        
        # Updated angles
        phi   = IMU_angles[0].item()
        theta = IMU_angles[1].item()
        psi   = IMU_angles[2].item()
        
        # Using the updated angles for the specific force
        theDCM = Body2Inertial (phi, theta, psi)
        
        """ 2.Transformation of specific force resolving axes  """
        specificForce_I = specificForceTransform(theDCM, BodyRates, dt)@specificForce_B
        
        """ 3.Velocity update """
        Pos_I = getPosition(states)
        
        # Earth's graviational field model (inertial frame)
        gravAccel = gravityField(Pos_I)
        
        effectiveAccel = specificForce_I + gravAccel
        
        Vel_I = getVelocity(states)
        
        Vel_I += effectiveAccel*dt
        
        """  4. Position Update  """
        Pos_I += Vel_I*dt
        
        # Add weights to the estimates
        predictedStates = states
        predictedStates[0] = phi
        predictedStates[1] = theta
        predictedStates[2] = psi
        predictedStates[3] = Vel_I[0].item()
        predictedStates[4] = Vel_I[1].item()
        predictedStates[5] = Vel_I[2].item()
        predictedStates[6] = Pos_I[0].item()
        predictedStates[7] = Pos_I[1].item()
        predictedStates[8] = Pos_I[2].item()
        
        # Unscented Transform
        muX += Wm[0,i] * predictedStates
        
    # Covariance Matrix    
    Px = myUKF.ukfCovariance(sigmas,muX,Wc,Qk)
    
    Px = 0.5*(Px + np.transpose(Px)) # Helpful for conserving symmetry
    
    # Creating new Sigma Points from new states
    myUKF.states = muX

    newSigmas = myUKF.sigmaPoints()
    
    # =================== Measurement Sigma points ===================
    measureSigmas = np.zeros((sensors,n))  # (sensors x 2n+1)
    
    muZ = np.zeros((sensors,1)) # empty vector for now
    
    for j in range (n):
        # Apply the measurement models using the sigma points
        states = newSigmas[:,j].reshape(m,1)
        
        currentAngles = getAngles(states)
        Vel_I = getVelocity(states)
        Pos_I = getPosition(states)
        
        phi   = currentAngles[0].item()
        theta = currentAngles[1].item()
        psi   = currentAngles[2].item()
        
        measureSigmas[0][j] = phi
        measureSigmas[1][j] = theta
        measureSigmas[2][j] = psi
        measureSigmas[3][j] = Vel_I[0].item()
        measureSigmas[4][j] = Vel_I[1].item()
        measureSigmas[5][j] = Vel_I[2].item()
        measureSigmas[6][j] = Pos_I[0].item()
        measureSigmas[7][j] = Pos_I[1].item()
        measureSigmas[8][j] = Pos_I[2].item()
        
        muZ += Wm[0,j] * measureSigmas[:,j].reshape(sensors,1)
        
    # Measurements
    if GPS_AVAILABLE:
        
        # Pseudo heading measurement with GPS data
        Vx = gps.velEast[index]
        Vy = gps.velNorth[index]
        gpsHeading = math.atan2(Vy,Vx)
        
        # Centripetal Acceleration
        r = np.sqrt(gps.posNorth[i]**2 + gps.posEast[i]**2)
        centripetalAccel = r*(BodyRates[2].item()**2)

        # Averaging the accelerometer measurements
        avgAccel = (1/count)*accelSum
        
        accelMag = np.linalg.norm(avgAccel)
        
        # Converting to a unit vector (not needed since the domain of inverse tangent is -inf to inf)
        ax = avgAccel[0].item() / accelMag
        ay = (avgAccel[1].item() - centripetalAccel)/ accelMag
        az = avgAccel[2].item() / accelMag
        
        imuRoll = math.atan2(ay, az)
        
        den = np.sqrt(ay**2 + az**2)
        
        imuPitch = math.atan2(ax, den)

        # Reset the sum & counter
        accelSum = np.zeros([3,1])
        count = 0

        zk = np.array([[imuRoll],   
                       [imuPitch],   
                       [gpsHeading],   
                       [gps.velEast[index]],
                       [gps.velNorth[index]],
                       [gps.velUp[index]],
                       [gps.posEast[index]],
                       [gps.posNorth[index]],
                       [gps.posUp[index]]])
        
        # Increment the GPS data index
        index += 1
    else:
        # Make the necessary modification assuming GPS isn't available 
        # Extrapolate the old measurements
        
        gpsExtrap = extrapolateMeasurement(muX, inputAccel)
        
        # Pseudo heading measurement with GPS data
        Vx = gpsExtrap[1].item()
        Vy = gpsExtrap[3].item()
        gpsHeading = math.atan2(Vy,Vx)
        
        # Centripetal Acceleration
        r = np.sqrt(gpsExtrap[0].item()**2 + gpsExtrap[2].item()**2)
        centripetalAccel = r*(BodyRates[2].item()**2)

        # Creating a running sum of accelerometer measurements (to be averaged) in between GPS measurements
        count += 1
        
        accelSum += np.array([[specificForce_B[0].item()],
                              [specificForce_B[1].item()],
                              [specificForce_B[2].item()]])
        
        accelMag = np.linalg.norm(specificForce_B)
        
        # Converting to a unit vector
        ax = specificForce_B[0].item() / accelMag
        ay = (specificForce_B[1].item()-centripetalAccel) / accelMag
        az = specificForce_B[2].item() / accelMag
        
        imuRoll = math.atan2(ay, az)
        
        den = np.sqrt(ay**2 + az**2)
        
        imuPitch = math.atan2(ax, den)
        
        zk = np.array([[imuRoll],        
                       [imuPitch],        
                       [gpsHeading],       
                       [gpsExtrap[1].item()],    
                       [gpsExtrap[3].item()],    
                       [gpsExtrap[5].item()],  
                       [gpsExtrap[0].item()],    
                       [gpsExtrap[2].item()],   
                       [gpsExtrap[4].item()]])        
        
    # Residual 
    yk = zk - muZ
        
    # Measurement Model Covariance    
    Pz = myUKF.ukfCovariance(measureSigmas,muZ,Wc,Rk)
    
    """ Adpative process noise """
    if adpativeProcessNoise:
        for i in range(len(yk)):
            # Innovation uncertainty
            Theory = np.sqrt(Pz[i][i])
            if np.abs(yk[i]) > 1*Theory:
                Qk[i][i] += 0.00010
    
    # Cross Covariance
    Pxz = myUKF.crossCovariance(newSigmas,measureSigmas,muX,muZ, Wc)
    
    # =================== Kalman Gain ===================
    K = Pxz@inv(Pz)
    
    # Estimation
    xk = muX + (K@yk)
    
    # Covariance Update
    Pk = myUKF.CovarianceUpdate(Px, Pz, K)
    
    Pk = 0.5*(Pk + np.transpose(Pk)) # Helpful for conserving symmetry
    
    #============= store for plotting =============
    
    # Similarity transforms to easily extract certain states
    temp  = getPosVel(xk)
    temp2 = getAngles(xk)
    temp3 = getBias(xk)
    
    posEast[idx]  = temp[0].item()
    posNorth[idx] = temp[2].item()
    posUp[idx]    = temp[4].item()
    velEast[idx]  = temp[1].item()
    velNorth[idx] = temp[3].item()
    velUp[idx]    = temp[5].item()
    roll[idx]     = np.rad2deg(temp2[0].item())
    pitch[idx]    = np.rad2deg(temp2[1].item())
    yaw[idx]      = np.rad2deg(temp2[2].item())
    biasAx[idx]   = temp3[0].item()
    biasAy[idx]   = temp3[1].item()
    biasAz[idx]   = temp3[2].item()
    biasGx[idx]   = temp3[3].item()
    biasGy[idx]   = temp3[4].item()
    biasGz[idx]   = temp3[5].item()
    
    with open(tmFileName, "a") as tmFile:
        tmFile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(truth.time[idx], roll[idx], pitch[idx] , yaw[idx] , velEast[idx], velNorth[idx], velUp[idx], posEast[idx], posNorth[idx], posUp[idx]))
    
    # Redefining for next iteration
    myUKF.states = xk
    
    # Is the covariance matrix positive definite?
    test = np.all(np.linalg.eigvals(Pk) > 0)
    
    if test == False:
        # Need to get rid of the negative eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(Pk)
        
        # replace the negative eigenvalues with 0 (or a small positive number as Brissette et al. 2007 suggest)
        for i in range(len(eigenvalues)):
            if eigenvalues[i] < 0:
                eigenvalues[i] = 0.5
                
        # recompose the covariance matrix, now it should be positive definite.
        # vectors* values * vectors.'
        # cov = v*d*v';
        Pk = (eigenvectors@np.diag(eigenvalues))@np.transpose(eigenvectors)
        
    myUKF.covariance = Pk
    
        
    # Storing Residuals    
    x1Residual[idx] = truth.roll[idx]     - np.rad2deg(temp2[0].item()) 
    x2Residual[idx] = truth.pitch[idx]    - np.rad2deg(temp2[1].item())
    x3Residual[idx] = truth.yaw[idx]      - np.rad2deg(temp2[2].item())
    x4Residual[idx] = truth.velEast[idx]  - temp[1].item()
    x5Residual[idx] = truth.velNorth[idx] - temp[3].item()
    x6Residual[idx] = truth.velUp[idx]    - temp[5].item()
    x7Residual[idx] = truth.posEast[idx]  - temp[0].item()
    x8Residual[idx] = truth.posNorth[idx] - temp[2].item()
    x9Residual[idx] = truth.posUp[idx]    - temp[4].item()
    
    # Storting Uncertainties
    x1std[idx]       =  np.rad2deg(np.sqrt(Pk[0][0]))
    x2std[idx]       =  np.rad2deg(np.sqrt(Pk[1][1]))
    x3std[idx]       =  np.rad2deg(np.sqrt(Pk[2][2]))
    x4std[idx]       =  np.sqrt(Pk[3][3]) 
    x5std[idx]       =  np.sqrt(Pk[4][4]) 
    x6std[idx]       =  np.sqrt(Pk[5][5]) 
    x7std[idx]       =  np.sqrt(Pk[6][6]) 
    x8std[idx]       =  np.sqrt(Pk[7][7]) 
    x9std[idx]       =  np.sqrt(Pk[8][8]) 
    
    if adjustForBias:
        gyroBias = np.array([[biasGx[idx]],
                              [biasGy[idx]],
                              [biasGz[idx]]]) 
        
        accelBias = np.array([[biasAx[idx]],
                              [biasAy[idx]],
                              [biasAz[idx]]])

endTime = time.time()
runTime = endTime - startTime

# Mean Error
meanRollError  = stats.mean(x1Residual)
meanPitchError = stats.mean(x2Residual)
meanYawError   = stats.mean(x3Residual)
meanVxError    = stats.mean(x4Residual)
meanVyError    = stats.mean(x5Residual)
meanVzError    = stats.mean(x6Residual)
meanPxError    = stats.mean(x7Residual)
meanPyError    = stats.mean(x8Residual)
meanPzError    = stats.mean(x9Residual)

# Standard deviation
stdRollError  = stats.stdev(x1Residual)
stdPitchError = stats.stdev(x2Residual)
stdYawError   = stats.stdev(x3Residual)
stdVxError    = stats.stdev(x4Residual)
stdVyError    = stats.stdev(x5Residual)
stdVzError    = stats.stdev(x6Residual)
stdPxError    = stats.stdev(x7Residual)
stdPyError    = stats.stdev(x8Residual)
stdPzError    = stats.stdev(x9Residual)


print("\nComplete, t= {}s\n".format(runTime))

print("Plotting Data...\n")
        
# ============= Plotting States & Estimates =============
plt.figure(1)
plt.plot(truth.time, posEast, 'r--', label="Estimate")
plt.plot(truth.time, truth.posEast, label="Truth")
plt.xlabel("Time [s]")
plt.ylabel("East Position [m]")
plt.title("Ground Vehicle Position (East)")
plt.legend()
plt.grid()
plt.savefig(savePath + "\Ground_Vehicle_Position_East.png") 
plt.show()

plt.figure(2)
plt.plot(truth.time, posNorth, 'r--', label="Estimate")
plt.plot(truth.time, truth.posNorth, label="Truth")
plt.xlabel("Time [s]")
plt.ylabel("North Position [m]")
plt.title("Ground Vehicle Position (North)")
plt.legend()
plt.grid()
plt.savefig(savePath + "\Ground_Vehicle_Position_North.png") 
plt.show()

plt.figure(3)
plt.plot(truth.time, posUp, 'r--', label="Estimate")
plt.plot(truth.time, truth.posUp, label="Truth")
plt.xlabel("Time [s]")
plt.ylabel("Up Position [m]")
plt.title("Ground Vehicle Position (Up)")
plt.legend()
plt.ylim([-5, 5])
plt.grid()
plt.savefig(savePath + "\Ground_Vehicle_Position_Up.png") 
plt.show()

plt.figure(4)
plt.plot(posEast, posNorth, 'r--', label="Estimate")
plt.plot(truth.posEast, truth.posNorth, label="Truth")
plt.xlabel("East Position [m]")
plt.ylabel("North Position [m]")
plt.title("Ground Vehicle Horizontal Position")
plt.legend()
plt.grid()
plt.savefig(savePath + "\Ground_Vehicle_Horizontal_Position.png") 
plt.show()

plt.figure(5)
plt.plot(truth.time, velEast, 'r--', label="Estimate")
plt.plot(truth.time, truth.velEast, label="Truth")
plt.xlabel("Time [s]")
plt.ylabel("East Velocity [m/s]")
plt.title("Ground Vehicle Velocity (East)")
plt.ylim([-5, 5])
plt.legend()
plt.grid()
plt.savefig(savePath + "\Ground_Vehicle_Velocity_East.png") 
plt.show()

plt.figure(6)
plt.plot(truth.time, velNorth, 'r--', label="Estimate")
plt.plot(truth.time, truth.velNorth, label="Truth")
plt.xlabel("Time [s]")
plt.ylabel("North Velocity [m/s]")
plt.title("Ground Vehicle Velocity (North)")
plt.ylim([-5, 5])
plt.legend()
plt.grid()
plt.savefig(savePath + "\Ground_Vehicle_Velocity_North.png") 
plt.show()

plt.figure(7)
plt.plot(truth.time, velUp, 'r--', label="Estimate")
plt.plot(truth.time, truth.velUp, label="Truth")
plt.xlabel("Time [s]")
plt.ylabel("Up Velocity [m/s]")
plt.title("Ground Vehicle Velocity (Up)")
plt.ylim([-5, 5])
plt.legend()
plt.grid()
plt.savefig(savePath + "\Ground_Vehicle_Velocity_Up.png") 
plt.show()

plt.figure(8)
plt.plot(velEast, velNorth, 'r--', label="Estimate")
plt.plot(truth.velEast, truth.velNorth, label="Truth")
plt.xlabel("East Velocity [m/s]")
plt.ylabel("North Velocity [m/s]")
plt.title("Ground Vehicle Horizontal Velocity")
plt.ylim([-5, 5])
plt.legend()
plt.grid()
plt.savefig(savePath + "\Ground_Vehicle_Horizontal_Velocity.png") 
plt.show()

plt.figure(9)
plt.plot(truth.time, roll, 'r--', label="Estimate")
plt.plot(truth.time, truth.roll, label="Truth")
plt.xlabel("Time [s]")
plt.ylabel("Roll Angle [deg]")
plt.title("Roll Angle")
plt.ylim([-1, 8])
plt.legend()
plt.grid()
plt.savefig(savePath + "\Roll_Angle.png") 
plt.show()

plt.figure(10)
plt.plot(truth.time, pitch, 'r--', label="Estimate")
plt.plot(truth.time, truth.pitch, label="Truth")
plt.xlabel("Time [s]")
plt.ylabel("Pitch Angle [deg]")
plt.title("Pitch Angle")
plt.ylim([-1, 1])
plt.legend()
plt.grid()
plt.savefig(savePath + "\Pitch_Angle.png") 
plt.show()

plt.figure(11)
plt.plot(truth.time, yaw, 'r--', label="Estimate")
plt.plot(truth.time, truth.yaw, label="Truth")
plt.xlabel("Time [s]")
plt.ylabel("Yaw Angle [deg]")
plt.title("Yaw Angle")
plt.ylim([-180, 180])
plt.legend()
plt.grid()
plt.savefig(savePath + "\Yaw_Angle.png") 
plt.show()

# ============= Plotting Residual and Uncertainty =============
if plotUncertainty:
    plt.figure(12)
    plt.plot(truth.time, [numSigmas * i for i in x1std],'b', linewidth = 2)
    plt.plot(truth.time,x1Residual, 'r--', label = '[True - Estimated]')
    plt.plot(truth.time, [-numSigmas * i for i in x1std],'b', linewidth = 2)
    plt.ylabel("Errors in Roll [deg]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("{} sigma angle residual".format(numSigmas))
    plt.grid()
    plt.savefig(savePath + "\Roll_Uncertanity.png") 
    plt.show()    
    
    plt.figure(13)
    plt.plot(truth.time, [numSigmas * i for i in x2std],'b', linewidth = 2)
    plt.plot(truth.time,x2Residual, 'r--', label = '[True - Estimated]')
    plt.plot(truth.time, [-numSigmas * i for i in x2std],'b', linewidth = 2)
    plt.ylabel("Errors in Pitch [deg]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("{} sigma angle residual".format(numSigmas))
    plt.grid()
    plt.savefig(savePath + "\Pitch_Uncertanity.png") 
    plt.show()    
    
    plt.figure(14)
    plt.plot(truth.time, [numSigmas * i for i in x3std],'b', linewidth = 2)
    plt.plot(truth.time,x3Residual, 'r--', label = '[True - Estimated]')
    plt.plot(truth.time, [-numSigmas * i for i in x3std],'b', linewidth = 2)
    plt.ylabel("Errors in Yaw [deg]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("{} sigma angle residual".format(numSigmas))
    plt.grid()
    plt.savefig(savePath + "\Yaw_Uncertanity.png") 
    plt.show()    
    
    plt.figure(15)
    plt.plot(truth.time, [numSigmas * i for i in x4std],'b', linewidth = 2)
    plt.plot(truth.time,x4Residual, 'r--', label = '[True - Estimated]')
    plt.plot(truth.time, [-numSigmas * i for i in x4std],'b', linewidth = 2)
    plt.ylabel("Errors in Velocity (East) [m/s]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("{} sigma velocity residual".format(numSigmas))
    plt.grid()
    plt.savefig(savePath + "\Velocity_East_Uncertanity.png") 
    plt.show()   
    
    plt.figure(16)
    plt.plot(truth.time, [numSigmas * i for i in x5std],'b', linewidth = 2)
    plt.plot(truth.time,x5Residual, 'r--', label = '[True - Estimated]')
    plt.plot(truth.time, [-numSigmas * i for i in x5std],'b', linewidth = 2)
    plt.ylabel("Errors in Velocity (North) [m/s]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("{} sigma velocity residual".format(numSigmas))
    plt.grid()
    plt.savefig(savePath + "\Velocity_North_Uncertanity.png") 
    plt.show() 
    
    plt.figure(17)
    plt.plot(truth.time, [numSigmas * i for i in x6std],'b', linewidth = 2)
    plt.plot(truth.time,x6Residual, 'r--', label = '[True - Estimated]')
    plt.plot(truth.time, [-numSigmas * i for i in x6std],'b', linewidth = 2)
    plt.ylabel("Errors in Velocity (Up) [m/s]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("{} sigma velocity residual".format(numSigmas))
    plt.grid()
    plt.savefig(savePath + "\Velocity_Up_Uncertanity.png") 
    plt.show() 
    
    plt.figure(18)
    plt.plot(truth.time, [numSigmas * i for i in x7std],'b', linewidth = 2)
    plt.plot(truth.time,x7Residual, 'r--', label = '[True - Estimated]')
    plt.plot(truth.time, [-numSigmas * i for i in x7std],'b', linewidth = 2)
    plt.ylabel("Errors in Position (East) [m]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("{} sigma position residual".format(numSigmas))
    plt.grid()
    plt.savefig(savePath + "\Position_East_Uncertanity.png") 
    plt.show() 
    
    plt.figure(19)
    plt.plot(truth.time, [numSigmas * i for i in x8std],'b', linewidth = 2)
    plt.plot(truth.time,x8Residual, 'r--', label = '[True - Estimated]')
    plt.plot(truth.time, [-numSigmas * i for i in x8std],'b', linewidth = 2)
    plt.ylabel("Errors in Position (North) [m]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("{} sigma position residual".format(numSigmas))
    plt.grid()
    plt.savefig(savePath + "\Position_North_Uncertanity.png") 
    plt.show() 
    
    plt.figure(20)
    plt.plot(truth.time, [numSigmas * i for i in x9std],'b', linewidth = 2)
    plt.plot(truth.time,x9Residual, 'r--', label = '[True - Estimated]')
    plt.plot(truth.time, [-numSigmas * i for i in x9std],'b', linewidth = 2)
    plt.ylabel("Errors in Position (Up) [m]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("{} sigma position residual".format(numSigmas))
    plt.grid()
    plt.savefig(savePath + "\Position_Up_Uncertanity.png") 
    plt.show() 

# ============= Plotting IMU Bias =============

# Accelerometer
plt.figure(21)
plt.plot(truth.time, biasAx)
plt.xlabel("Time [s]")
plt.ylabel("Accel Bias X")
plt.title("Accelerometer Bias (X)")
plt.grid()
plt.savefig(savePath + "\Accelerometer_Bias_X.png") 
plt.show()

plt.figure(22)
plt.plot(truth.time, biasAy)
plt.xlabel("Time [s]")
plt.ylabel("Accel Bias Y")
plt.title("Accelerometer Bias (Y)")
plt.grid()
plt.savefig(savePath + "\Accelerometer_Bias_Y.png") 
plt.show()

plt.figure(23)
plt.plot(truth.time, biasAz)
plt.xlabel("Time [s]")
plt.ylabel("Accel Bias Z")
plt.title("Accelerometer Bias (Z)")
plt.grid()
plt.savefig(savePath + "\Accelerometer_Bias_Z.png") 
plt.show()

# Gyroscope
plt.figure(24)
plt.plot(truth.time, biasGx)
plt.xlabel("Time [s]")
plt.ylabel("Gyro Bias X")
plt.title("Gyroscope Bias (X)")
plt.grid()
plt.savefig(savePath + "\Gyroscope_Bias_X.png") 
plt.show()

plt.figure(25)
plt.plot(truth.time, biasGy)
plt.xlabel("Time [s]")
plt.ylabel("Gyro Bias Y")
plt.title("Gyroscope Bias (Y)")
plt.grid()
plt.savefig(savePath + "\Gyroscope_Bias_Y.png") 
plt.show()

plt.figure(26)
plt.plot(truth.time, biasGz)
plt.xlabel("Time [s]")
plt.ylabel("Gyro Bias Z")
plt.title("Gyroscope Bias (Z)")
plt.grid()
plt.savefig(savePath + "\Gyroscope_Bias_Z.png") 
plt.show()

print("Saving plots...\n")

print("Writing the average errors to a .txt file...")
with open(fileName, "w") as writeFile:
    writeFile.write("\n======= Printing the average state errors =======\n")
    writeFile.write("Average roll error           = {} deg\n".format(meanRollError))
    writeFile.write("Average pitch error          = {} deg\n".format(meanPitchError))
    writeFile.write("Average yaw error            = {} deg\n".format(meanYawError))
    writeFile.write("Average east velocity error  = {} m/s\n".format(meanVxError))
    writeFile.write("Average north velocity error = {} m/s\n".format(meanVyError))
    writeFile.write("Average up velocity error    = {} m/s\n".format(meanVzError))
    writeFile.write("Average east position error  = {} m\n".format(meanPxError))
    writeFile.write("Average north position error = {} m\n".format(meanPyError))
    writeFile.write("Average up position error    = {} m\n".format(meanPzError))
    
print("\nComplete\n")

