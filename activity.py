import numpy as np
from health import Health, BikeHealth, SoccerHealth
from datetime import datetime
from file import File 

class Activity:
    def __init__(self, file):
        self.file = File(file)
        self.date = self.setDate(self.file.name)

    def setDate(self, file) -> datetime:
        # Current date and time. Should really decode the file.
        return datetime.now()
    
    def getHealth(self) -> Health:
        raise TypeError("Activity does not have Health Info")

class BikeActivity(Activity):
    def __init__(self, file):
        super().init(self, file)
        self.health = BikeHealth()
    
    def getHealth(self) -> Health:
        return self.health
    
    def setDistance(self, distance):
        self.distance = distance

class SoccerActivity(Activity):
    def __init__(self, file):
        super().init(self, file)
        self.health = SoccerHealth()
    
    def getHealth(self) -> Health:
        return self.health

class SoccerGame(SoccerActivity):
    def __init__(self, file):
        super().__init__(self,file)

    def setScore(self, our, theirs):
        self.our_score = our
        self.their_score = theirs 
    
    def setStats(self, goals, assists):
        self.goals = goals
        self.assists = assists 
    
    