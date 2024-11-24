from distutils.log import error
import string
import numpy as np
import os 
from garmin_fit_sdk import Decoder, Stream
from pathlib import Path 

class File(object):
    file_type = None 
    def __init__(self, name):
        self.name = name
        self.size = self.setSize()
        self.setValidFiles()
    
    @classmethod 
    def setValidFiles(self):
        self.valid_files = self.__subclasses__()

    def setSize(self):
        if os.path.isfile(self.name):
            self.size = os.path.getsize(self.name)
        return 
    
    def readData(self) -> string:
        raise TypeError("Unknown FileType, cannot read")

    def loadData(self, path_to_file, type_def=list(str())):
        file_type = Path(path_to_file).suffix[1:]
        if len(type_def) > 0:
            valid_file_suffix = type_def 
        else:     
            valid_file_suffix = [v.file_type for v in self.valid_files]

        if file_type in valid_file_suffix:
            idx = valid_file_suffix.index(file_type)
            file = self.valid_files[idx](path_to_file)
        else: 
            raise TypeError ("FileType not in valid FileTypes: " + ",".join(valid_file_suffix))
        return file 
    
    def __str__(self):
        return self.file_type



class GPX(File):
    file_type = "gpx" 
    def __init__(self, name):
        super().__init__(name)

    @classmethod 
    def setValidFiles(self):
        self.valid_files = [self]
    


    

    def readData(self): 
        stream = Stream.from_file(self.name)
        decoder = Decoder(stream)
        messages, errors = decoder.read()
        return messages, errors

class FIT(File):
    file_type = "fit" 
    def __init__(self, name):
        super().__init__(self,name)
    
    @classmethod 
    def setValidFiles(self):
        self.valid_files = [self]
        

class KML(File):
    file_type = "kml" 
    def __init__(self, name):
        super().__init__(self,name)
    
    @classmethod 
    def setValidFiles(self):
        self.valid_files = [self]
