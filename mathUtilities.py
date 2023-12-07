import numpy as np
import math
from numpy.linalg import matrix_power
import pandas as pd
from similarityTransform import getPosVel


###########################################################
#   Function    : stateData()
#
#   Description : Takes csv file data and outputs a numpy 
#                 array
#
#   Example call: GPSData = stateData("GPSMeasurements.csv")
############################################################
def stateData(fileName):
    # importing the state data
    dataframe = pd.read_csv(fileName, delimiter= ',')

    # converting the info from a pandas dataframe into a numpy array
    stateArray = dataframe.to_numpy()

    return stateArray

###########################################################
#   Function    : rotMat()
#
#   Description : Performs coordinate frame transformation
#                 the input angle is intented to be in radians
#
#   Example call: R = rotMat(0.4363, 'z')
############################################################

def rotMat(angle, axis):
    if axis == 'x' or axis == 'X':
        # Roll 
        R = np.array([[1, 0, 0],
                     [0, np.cos(angle), np.sin(angle)], 
                     [0, -np.sin(angle), np.cos(angle)]])
        
    elif axis == 'y' or axis == 'Y':
        # Pitch 
        R = np.array([[np.cos(angle), 0, -np.sin(angle)], 
                     [0, 1, 0],
                     [np.sin(angle), 0, np.cos(angle)]])
        
    elif axis == 'z' or axis == 'Z':
        # Yaw
        R = np.array([[np.cos(angle), np.sin(angle),  0], 
                     [-np.sin(angle), np.cos(angle), 0],
                     [0, 0 ,1]])
        
    return R



###########################################################
#   Function    : Inertial2Body()
#
#   Description : Performs a series of coordinate frame 
#                 transformations to convert from the inertial
#                 frame to the body frame. The input angles are
#                 intented to be in radians
#
#   Example call: R  = Inertial2Body(0.0873, 0.3840, -1.5533)
############################################################

def Inertial2Body(roll, pitch, yaw):
    # The @ means 'matrix multiply'
    return rotMat(roll,'x')@(rotMat(pitch, 'y')@rotMat(yaw, 'z'))



###########################################################
#   Function    : Body2Inertial()
#
#   Description : Performs a series of coordinate frame 
#                 transformations to convert from the body
#                 frame to the inertial frame. The input angles are
#                 intented to be in radians
#
#   Example call: R  = Body2Inertial(0.0873, 0.3840, -1.5533)
############################################################  
  
def Body2Inertial (roll, pitch, yaw):
    return np.transpose(Inertial2Body(roll, pitch, yaw))


###########################################################
#   Function    : extractEuler()
#
#   Description : Given an inertial to body frame transformation
#                 matrix, the Euler angles can be extracted using
#                 inverse trig functions on the values at certain 
#                 positions on the matrix
#
#   Example call: angles = extractEuler(matrix)
############################################################ 

def extractEuler(R):
    # Extracting Euler angles from Inertial2Body(roll, pitch, yaw)
    phi   = math.atan2(R[1,2],R[2,2])
    theta = math.asin(-R[0,2])
    psi   = math.atan2(R[0,1],R[0,0])
    
    return np.rad2deg(np.array([[phi],
                             [theta],
                             [psi]]))


###########################################################
#   Function    : EulerKinematic()
#
#   Description : Using Euler's Kinematical equations, the
#                 attitude rate from the dynamics can be 
#                 converted into attitude rates (roll rate,
#                 pitch rate, yaw rate). However if the 
#                 pitch angle reaches 90 degrees, there is
#                 a division by zero error
#
#                 Input angles should be an array size = 2
#                 Body rates should be a 3x1 column vector
#
#   Example call: angleRate = EulerKinematic(angles, BodyRates)
############################################################ 

def EulerKinematic(angles, BodyRate):
    
    roll  = angles[0].item()
    pitch = angles[1].item()
    
    # pitch == 90 -> Gimbal Lock
    # 1/cos(90) = undefined
    
    R = np.array([[1, np.sin(roll)*np.tan(pitch), np.cos(roll)*np.tan(pitch)], 
                  [0, np.cos(roll), -np.sin(roll)],
                  [0, np.sin(roll)/np.cos(pitch), np.cos(roll)/np.cos(pitch)]])
    
    # Body Rates [p,q,r].' from the gyroscope
    # The @ means 'matrix multiply'
    return R@BodyRate


###########################################################
#   Function    : IMU_AttitudeUpdate()
#
#   Description : Rather than using Euler's kinematical 
#                 equations, another way of extracting 
#                 Euler angles is by calculating the rate
#                 of the DCM, integrating, and extracting 
#                 the angles similar to extractEuler()
#
#   Example call: angles = IMU_AttitudeUpdate(dcmOLD, Gyro, dt)
############################################################ 

def IMU_AttitudeUpdate(dcmOLD, Gyro, dt):
    
    # Skew-symmetric matrix of the angular rate vector
    wx = Gyro[0].item()
    wy = Gyro[1].item()
    wz = Gyro[2].item()
    
    OMEGA = np.array([[0, -wz,  wy], 
                      [wz, 0,  -wx],
                      [-wy, wx ,0]])
    
    # Identity Matrix    
    I = np.identity(3)
        
    #incr = I + OMEGA*dt + (matrix_power(OMEGA*dt, 2)*(1/np.math.factorial(2))) + (matrix_power(OMEGA*dt, 3)*(1/np.math.factorial(3)))

    #dcmNEW = dcmOLD@incr

    alpha = OMEGA*dt
    
    alphaMag = np.linalg.norm(alpha)

    A = I + ((np.sin(alphaMag)/alphaMag)*alpha) + (((1-np.cos(alphaMag))/alphaMag**2) * matrix_power(alpha, 2))
    
    dcmNEW = dcmOLD@A
    
    # Extracting Euler angles from Body2Inertial(roll, pitch, yaw)
    phi   = math.atan2(dcmNEW[2,1],dcmNEW[2,2])
    theta = math.asin(-dcmNEW[2,0])
    psi   = math.atan2(dcmNEW[1,0],dcmNEW[0,0])
    
    return np.array([[phi],
                     [theta],
                     [psi]])

###########################################################
#   Function    : tiltAngles()
#
#   Description : Euler angles (roll & pitch) can be found
#                 by measuring acceleration relative to gravity
#
#                 Input angles acceleration should be a 3x1
#                 column vector
#
#   Example call: tiltAngles(accel)
############################################################ 

def tiltAngles(accel):
    
    # Creating a unit vector by dividing by the magnitude
    unit_accel = accel * (1/np.linalg.norm(accel))
    
    # Components of the unit vector
    ax = unit_accel[0].item()
    ay = unit_accel[1].item()
    az = unit_accel[2].item()
    
    # sqrt(ay^2 + az^2) = cos(pitch)
    temp = np.linalg.norm([ay, az])
    
    # Orientation using acceleration measurements
    roll  = math.atan2(ay, az)
    #pitch = math.atan2(-ax, temp) 
    pitch = math.atan2(ax, temp)
    
    return np.rad2deg(np.array([[roll],
                             [pitch]]))

def gravityField(pos):
    R0 = 6378137.0 # [m]
    
    # Geocentric Position
    x = pos[0].item()
    y = pos[1].item()
    z = pos[2].item() + R0
    
    pos = np.array([[x],
                    [y],
                    [z]])

    mag = np.linalg.norm(pos)

    x_u = x/mag
    y_u = y/mag
    z_u = z/mag

    # mu is the Earth's graviational constant. Using WGS-84 value
    mu = 3.986004418E14 # [m^3/s^2]

    # J2 is the Earth's second gravitational constant 
    J2 = 1.082627E-3 #

    first = -mu/(mag**2)
    second = 1.5*J2*(R0/mag)**2

    gravX = first * ((1 + second * (1 - 5*z_u**2)  ) * x_u)
    gravY = first * ((1 + second * (1 - 5*z_u**2)  ) * y_u)
    gravZ = first * ((1 + second * (3 - 5*z_u**2)  ) * z_u)

    return np.array([[gravX],
                     [gravY],
                     [gravZ]])

# def gravityField(pos):
#     #R0 = 6378137.0 # [m]
    
#     # Geocentric Position
#     #pos = R0 + pos

#     x = pos[0].item()
#     y = pos[1].item()
#     z = pos[2].item()    

#     # mu is the Earth's graviational constant. Using WGS-84 value
#     mu = 3.986004418E14 # [m^3/s^2]

#     mag = np.linalg.norm(pos)

#     r = mag**2

#     gravX = (-mu*x) / r**(1.5)
#     gravY = (-mu*y) / r**(1.5)
#     gravZ = (-mu*z) / r**(1.5)

#     return np.array([[gravX],
#                      [gravY],
#                      [gravZ]])

def specificForceTransform(dcmOLD, Gyro, dt):
    
    # Skew-symmetric matrix of the angular rate vector
    wx = Gyro[0].item()
    wy = Gyro[1].item()
    wz = Gyro[2].item()
    
    OMEGA = np.array([[0, -wz,  wy], 
                      [wz, 0,  -wx],
                      [-wy, wx ,0]])
    
    # Identity Matrix    
    I = np.identity(3)
        
    alpha = OMEGA*dt
    
    alphaMag = np.linalg.norm(alpha)

    #A = I + ((np.sin(alphaMag)/alphaMag)*alpha) + (((1-np.cos(alphaMag))/alphaMag**2) * matrix_power(alpha, 2))
    
    A = I + (((1-np.cos(alphaMag))/alphaMag**2)*alpha) + ((1/alphaMag**2)*(1- np.sin(alphaMag)/alphaMag)*matrix_power(alpha, 2))
    
    dcmNEW = dcmOLD@A

    
    return dcmNEW


def extrapolateMeasurement(states, inputs):
    
    # Extracting the position and velocity from the state vector
    states = getPosVel(states)
    
    # Sampling rate related variables
    fasterRate = 100 # [Hz]
    slowerRate = 10  # [Hz]
    rateRatio = int(fasterRate/slowerRate)
    increments = np.linspace(1/fasterRate, 1/slowerRate, rateRatio) #[dt's]
    
    muX = np.zeros([6,1])

    for i in range(rateRatio):
        
        dt = increments[i]
        
        F = np.array([[1, dt, 0,  0,  0,  0],
                      [0,  1, 0,  0,  0,  0],
                      [0,  0, 1, dt,  0,  0],
                      [0,  0, 0,  1,  0,  0],
                      [0,  0, 0,  0,  1, dt],
                      [0,  0, 0,  0,  0,  1]])
        
        Gk = np.array([[0.5*dt**2,          0,         0],
                       [       dt,          0,         0],
                       [        0,  0.5*dt**2,         0],
                       [        0,         dt,         0],
                       [        0,          0, 0.5*dt**2],
                       [        0,          0,        dt]])
        
        xnew = F@states + Gk@inputs
        
        muX += xnew
        
    return muX*(1/rateRatio)
        
def extrapolateMeasurement2(states, inputs):
    
    # Extracting the position and velocity from the state vector
    H = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0]])
    
    states = (H@states)
    
    # Sampling rate related variables
    fasterRate = 100 # [Hz]
    slowerRate = 10  # [Hz]
    rateRatio = int(fasterRate/slowerRate)
    increments = np.linspace(1/fasterRate, 1/slowerRate, rateRatio) #[dt's]
    
    muX = np.zeros([6,1])

    for i in range(rateRatio):
        
        dt = increments[i]
        
        F = np.array([[1, dt, 0,  0,  0,  0],
                      [0,  1, 0,  0,  0,  0],
                      [0,  0, 1, dt,  0,  0],
                      [0,  0, 0,  1,  0,  0],
                      [0,  0, 0,  0,  1, dt],
                      [0,  0, 0,  0,  0,  1]])
        
        Gk = np.array([[0.5*dt**2,          0,         0],
                       [       dt,          0,         0],
                       [        0,  0.5*dt**2,         0],
                       [        0,         dt,         0],
                       [        0,          0, 0.5*dt**2],
                       [        0,          0,        dt]])
        
        xnew = F@states + Gk@inputs
        
        muX += xnew
        
    return muX*(1/rateRatio)        