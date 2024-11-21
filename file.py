import numpy as np
import os 
from garmin_fit_sdk import Decoder, Stream
from activity import Activity

class File():
    def __init__(self, name):
        self.name = name
        self.size = self.setSize()
    
    def setSize(self):
        if os.path.isfile(self.name):
            self.size = os.path.getsize(self.name)
        return 
    
    def readData(self) -> Activity:
        raise TypeError("Unknown FileType, cannot read")

class GPX(File):
    def __init__(self, name):
        super().__init__(self,name)

    

    def readData(self): 
        stream = Stream.from_file(self.name)
        decoder = Decoder(stream)
        messages, errors = decoder.read()
        return messages, errors

class FIT(File):
    def __init__(self, name):
        super().__init__(self,name)