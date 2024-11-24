from distutils.log import error
import string
import numpy as np
import os 
from garmin_fit_sdk import Decoder, Stream
from pathlib import Path 

class File(object):
    file_type = 'unspecified'
    def __init__(self, name, file_type=None):
        self.name = name
        self.path = self.name
        self.size = self.setSize()
        self.setValidFiles()
            
        
    
    @classmethod 
    def setValidFiles(self):
        self.valid_files = self.__subclasses__()

    def setSize(self):
        if os.path.isfile(self.path):
            self.size = os.path.getsize(self.path)
        return 
    
    def readData(self) -> string:
        raise TypeError("Unknown FileType, cannot read")

    def createNewFileInstance(self):
        return self.getFileType()(self.path)

    def getFileType(self):
        file_type = Path(self.path).suffix[1:]
        valid_file_suffix = [v.file_type for v in self.valid_files]
        if file_type in valid_file_suffix:
            idx = valid_file_suffix.index(file_type)
            file = self.valid_files[idx]
        else: 
            raise TypeError ("FileType not in valid FileTypes: " + ",".join(valid_file_suffix))
        return file
    
    def __str__(self):
        return f'{self.file_type.upper()} file type: {self.name}'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'


class GPX(File):
    file_type = "gpx" 
    def __init__(self, name):
        super().__init__(name, self.getFileType())

    @classmethod 
    def setValidFiles(self):
        self.valid_files = [self]

    def getFileType(self):
        return GPX 

    def readData(self): 
        stream = Stream.from_file(self.name)
        decoder = Decoder(stream)
        messages, errors = decoder.read()
        return messages, errors
    

class FIT(File):
    file_type = "fit" 
    def __init__(self, name):
        super().__init__(name, self.getFileType())
    
    @classmethod 
    def setValidFiles(self):
        self.valid_files = [self]

    def getFileType(self):
        return FIT 
    def readData(self): 
        stream = Stream.from_file(self.name)
        decoder = Decoder(stream)
        messages, errors = decoder.read()
        return messages, errors


class KML(File):
    file_type = "kml" 
    def __init__(self, name):
        super().__init__(name, self.getFileType())
    
    @classmethod 
    def setValidFiles(self):
        self.valid_files = [self]

    def getFileType(self):
        return KML  
    