from cmath import nan
import numpy as np
from activity import BikeActivity, SoccerActivity
import os 
from file import File 


class History:
    def __init__(self):
        self.BikeRides = list[BikeActivity]
        self.SoccerActivities = list[SoccerActivity]

    def getSoccerActivites(self):
        return self.SoccerActivites
    
    def getBikeRides(self):
        return self.BikeRides
    
    def loadDataFromPath(self, path, recursive=False):
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            if os.path.isfile(full_path):
                self.loadData(item)
            elif os.path.isdir(full_path):
                if recursive:
                    self.loadDataFromPath(full_path, recursive=True)
    def loadData(self, file):
        f = File(file)
        f = f.createNewFileInstance()
        a =f.readData()
        print(a)
                
                

    
                
