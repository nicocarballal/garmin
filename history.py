from cmath import nan
import numpy as np
from activity import BikeActivity, SoccerActivity
import os 


class History:
    def __init__(self):
        self.BikeRides = list[BikeActivity]
        self.SoccerActivities = list[SoccerActivity]

    def getSoccerActivites(self):
        return self.getSoccerActivites
    
    def getBikeRides(self):
        return self.BikeRides
    
    def loadDataFromPath(self, path, recursive=False):
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            if os.path.isfile(full_path):
                print(f"{item} is a directory")
                self.loadData(full_path)
            elif os.path.isdir(full_path):
                if recursive:
                    self.loadDataFromPath(full_path)
                

    
                
