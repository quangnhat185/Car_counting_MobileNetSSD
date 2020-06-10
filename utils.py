# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:29:03 2020

@author: Quang

"""

from scipy.spatial import distance as dist
from collections import OrderedDict
from json_minify import json_minify
import json
import numpy as np

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        """
        initialize the next unique object ID along with two ordered
        dictionaries used to keep track of mapping a given object ID
        to its centroid and number of consecutive frames it was been 
        masrked as "disappeared", respectively.
        """
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        
        """
        store the number of maximum consecutive frames a given 
        object is allowed to be marked as "disappeared" until
        we need to deregister the object from tracking
        """
        self.maxDisappeared = maxDisappeared
        
        """
        store the maximum distance between centroids to associate
        an object -- if the distance is larger than this maximum 
        distance we'll start to mark the object as "disappeared".
        """
        self.maxDistance = maxDistance
        
    def register(self, centroid):
        """
        when registering an object we use the next available object
        ID to store the centroid
        """
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        
    def deregister(self, objectID):
        """
        to deregister an object ID we delete the object ID
        from both of our respective dictionaries
        """
        del self.objects[objectID]
        del self.disappeared[objectID]
        
    def update(self, rects):
        """
        check to see if the list of input bounding box rectangels
        is empty
        """
        if len(rects) == 0:
            """
            loop over any existing tracked objects and mark 
            them as disappeared
            """
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] +=1
                
                """
                if we have reached a maximum number of consecutive 
                frames where a given object has been marked as missing,
                degister it.
                """
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            
            """
            return early as there are no centroid or tracking info
            to update
            """
            return self.objects
        
        # initialized an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects),2),dtype="int")
        
        # loop over the bounding box rectangels
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX,cY)
    
            """
            if we are currently not tracking any objects, take the input
            centroids and register each of them
            """
        if len(self.objects) == 0:
            for i in range(0,len(inputCentroids)):
                self.register(inputCentroids[i])
            """
            otherwise, they are currently tracking objects so we need to 
            try to match the input centroids to exisiting object centroids
            """
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            """
            compute the distance between each pair of object centroids
            and input centroids, respectively -- our goal will be to 
            match an input centroid to an existing object centroid
            """
            D = dist.cdist(np.array(objectCentroids), inputCentroids) 
            
            """
            in order to perform this matching, we must (1) find the 
            smallest value in each row and then (2) sort the row indexes 
            based on their minimum value so that the row with the smallest
            value as at the *front* of the index list
            """
            rows = D.min(axis=1).argsort()
            
            """
            next, we perform a similar process on the columns by 
            finding the smallest value in each column and then
            sorting using the previously computed row index list
            """
            cols = D.argmin(axis=1)[rows]
            
            """
            in order to determine if we need to update, register or 
            deregister an object we need to keep track of which of 
            the rows and column indexes we have already examined            
            """
            usedRows = set()
            usedCols = set()
            
            # loop over the combination of the (row, column) index tuples
            for (row,col) in zip(rows, cols):
                # if we have already examined either the row
                # orthe column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue
                
                """
                if the distance between centroids is greater than the 
                maximum distance, do not associate the two centroids 
                to the same objects
                """
                if D[row, col] > self.maxDistance:
                    continue
                
                """
                otherwise, grab the object ID for the current row,
                set its new centroid, and reset the disappeared counter
                """
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                
                """
                indicate that we have examined each of the row and 
                column indexes, respectively
                """
                usedRows.add(row)
                usedCols.add(col)
                
            # compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0,D.shape[0])).difference(usedRows)
            unusedCols = set(range(0,D.shape[1])).difference(usedCols)
            
            """
            in the event that the number of object centroids is equal or 
            greater than the number of input centroids, we need to check 
            and see if some of these objects have potentially disappeared
            """
            
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    """
                    grab the object ID for the corresponding row index
                    and increment the disappeared counter
                    """
                    objectID  = objectIDs[row]
                    self.disappeared[objectID] += 1
                    
                    """
                    check to see if the number of consecutive frames
                    which the object has been marked "disappeared" 
                    for warrants deregistering the object
                    """
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                        
                """
                otherwise, if the number of input centroids is greater
                than the number of existing object centroids we need to
                register each new input centroid as a trackable object
                """
            else: 
                for col in unusedCols:
                    self.register(inputCentroids[col])
                    
        # return the set of trackable objects
        return self.objects    

class TrackableObject:    
    def __init__(self, objectID, centroid):
        """
        store the objectID, then initialize a list of centroids
        using the current centroid
        """
        self.objectID = objectID
        self.centroids = [centroid]
        
        """
        initialize the dictionaries to store the timestamp and 
        position of the object at various point
        """
        self.timestamp = {"A": 0, "B": 0, "C": 0, "D": 0}            
        self.position = {"A": None, "B": None, "C": None, "D": None}
        self.lastPoint = False
        
        # initialise the object speeds in MPH and KMPH
        self.speedMPH = None
        self.speedKMPH = None
        
        """
        initialize two boolean, (1) used to indicate if the object's speed
        has already been estimated or not, and (2) used to indivate 
        if the object's speed has been logged or not
        """
        self.estimate = False
        self.loggd = False
        
        # initialize the direction of the object
        self.direction = None
        
    def calculate_speed(self, estimatedSpeeds):
        # calculate speed in KMPH and MPH
        self.speedKMPH = np.average(estimatedSpeeds)
        MILES_PER_ONE_KILOMETER = 0.621371
        self.speedMPH = self.speedKMPH * MILES_PER_ONE_KILOMETER

class Conf:
    def __init__(self, confPath):
        """
        load and store the configuration and update the 
        object't dictinonary
        """
        conf = json.loads(json_minify(open(confPath).read()))
        self.__dict__.update(conf)
    def __getitem__(self,k):
            # return the value associate with the supplied key
            return self.__dict__.get(k, None)
        