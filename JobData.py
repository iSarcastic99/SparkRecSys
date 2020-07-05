import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import numpy as np

class JobData:

    jobID_to_name = {}
    name_to_jobID = {}
    ratingsPath = 'jobData/rating.csv'
    jobsPath = 'jobData/editset.csv'
    
    def LoadJobData(self):

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        ratingsDataset = 0
        self.jobID_to_name = {}
        self.name_to_jobID = {}

        reader = Reader(line_format='user item rating', sep=',', skip_lines=1)

        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)

        with open(self.jobsPath, newline='', encoding='ISO-8859-1') as csvfile:
                jobReader = csv.reader(csvfile)
                next(jobReader)  #Skip header line
                for row in jobReader:
                    jobID = int(row[0])
                    jobName = row[1]
                    self.jobID_to_name[jobID] = jobName
                    self.name_to_jobID[jobName] = jobID

        return ratingsDataset

    def getUserRatings(self, user):
        userRatings = []
        hitUser = False
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                userID = int(row[0])
                if (user == userID):
                    jobID = int(row[1])
                    rating = float(row[2])
                    userRatings.append((jobID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break

        return userRatings

    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                jobID = int(row[1])
                ratings[jobID] += 1
        rank = 1
        for jobID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[jobID] = rank
            rank += 1
        return rankings
    
    def getGenres(self):
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0
        with open(self.jobsPath, newline='', encoding='ISO-8859-1') as csvfile:
            jobReader = csv.reader(csvfile)
            next(jobReader)  #Skip header line
            for row in jobReader:
                jobID = int(row[0])
                genreList = row[2].split('|')
                genreIDList = []
                for genre in genreList:
                    if genre in genreIDs:
                        genreID = genreIDs[genre]
                    else:
                        genreID = maxGenreID
                        genreIDs[genre] = genreID
                        maxGenreID += 1
                    genreIDList.append(genreID)
                genres[jobID] = genreIDList
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (jobID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[jobID] = bitfield
        
        return genres
    
    def getYears(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.jobsPath, newline='', encoding='ISO-8859-1') as csvfile:
            jobReader = csv.reader(csvfile)
            next(jobReader)
            for row in jobReader:
                jobID = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1)
                if year:
                    years[jobID] = int(year)
        return years
    
    def getMiseEnScene(self):
        mes = defaultdict(list)
        with open("LLVisualFeatures13K_Log.csv", newline='') as csvfile:
            mesReader = csv.reader(csvfile)
            next(mesReader)
            for row in mesReader:
                jobID = int(row[0])
                avgShotLength = float(row[1])
                meanColorVariance = float(row[2])
                stddevColorVariance = float(row[3])
                meanMotion = float(row[4])
                stddevMotion = float(row[5])
                meanLightingKey = float(row[6])
                numShots = float(row[7])
                mes[jobID] = [avgShotLength, meanColorVariance, stddevColorVariance,
                              meanMotion, stddevMotion, meanLightingKey, numShots]
        return mes
    
    def getJobName(self, jobID):
        if jobID in self.jobID_to_name:
            return self.jobID_to_name[jobID]
        else:
            return ""
        
    def getJobID(self, jobName):
        if jobName in self.name_to_jobID:
            return self.name_to_jobID[jobName]
        else:
            return 0