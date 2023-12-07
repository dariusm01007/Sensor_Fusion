from mathUtilities import stateData
import numpy as np

IMUData = stateData("IMUMeasurements.csv")
class IMU:
    def __init__(self):
        self.time    = IMUData[:,0] # [s]
        self.accelX  = IMUData[:,1] # [m/s^2]
        self.accelY  = IMUData[:,2] # [m/s^2]
        self.accelZ  = IMUData[:,3] # [m/s^2]
        self.omegaX  = IMUData[:,4] # [rad/s]
        self.omegaY  = IMUData[:,5] # [rad/s]
        self.omegaZ  = IMUData[:,6] # [rad/s]

    def dataRate(self):
        # Determining the update rate for the IMU
        time_increments = np.zeros(len(self.time))
        for i in range (0, len(self.time) - 1):
            time_increments[i] = self.time[i+1] - self.time[i]

        return 1 / np.mean(time_increments) # [Hz] 


GPSData = stateData("GPSMeasurements.csv")
class GPS:
    def __init__(self):
        self.time     = GPSData[:,0] # [s]
        self.posEast  = GPSData[:,1] # [m]
        self.posNorth = GPSData[:,2] # [m]
        self.posUp    = GPSData[:,3] # [m]
        self.velEast  = GPSData[:,4] # [m/s]
        self.velNorth = GPSData[:,5] # [m/s]
        self.velUp    = GPSData[:,6] # [m/s]

    def dataRate(self):
        # Determining the update rate for the IMU
        time_increments = np.zeros(len(self.time))
        for i in range (0, len(self.time) - 1):
            time_increments[i] = self.time[i+1] - self.time[i]

        return 1 / np.mean(time_increments) # [Hz] 

TruthData = stateData("TruthData.csv")
class Truth:
    def __init__(self):
        self.time     = TruthData[:,0]
        self.posEast  = TruthData[:,1]
        self.posNorth = TruthData[:,2]
        self.posUp    = TruthData[:,3]
        self.velEast  = TruthData[:,4]
        self.velNorth = TruthData[:,5]
        self.velUp    = TruthData[:,6]
        self.yaw      = np.rad2deg(TruthData[:,7])  
        self.pitch    = np.rad2deg(TruthData[:,8])
        self.roll     = np.rad2deg(TruthData[:,9])
