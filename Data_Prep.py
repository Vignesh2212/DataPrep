# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 18:44:30 2017

@author: Vignesh
"""
#Augmentation suggestions:
    #Check missing value NaN, None handling
    #Export to pdf, image
    #Bivariate group by variable - train, test, valid
    #Check input variable list exists in the dataframe


import pandas
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
class bivarGen(object):
    def __init__(self):
        #Set initial values. All sanity check flags are reset here
        self.preserveContDist = 0
        self.inpSanityCheck = 0
        self.DSName = "Input Data"
        self.DFCreated = 0
        self.overWrite = 0
        self.numBins = 10
        self.quantileCut = 0.05
        self.plotOneTgtClass = 0
        self.maxCatCutoff = 30
        self.cwdConfimation = 0
        self.bivarKeepMissing = 0

    #Constructor for setting some name about the dataset.
    def setDSName(self,DSName):
        self.DSName = DSName

    #Set project folder for storing bivariate pdfs to avoid clutter
    def setCurrentDir(self,newDir):
        self.oldDir = os.getcwd()
        os.chdir(newDir)

    #Use this method to revert working directory to old one, if available
    def revertDir(self):
        if self.oldDir != "":
            os.chdir(self.oldDir)
        else:
            print("No old working directory available to revert to")

    #By default, input dataset is overwrite protected.
    #call this constructor to overwrite any dataset.
    #Once a dataset is overwritten, it is overwrite protected again
    def setOverWrite(self, overWrite):
        if type(overWrite) is not int and (overWrite != 0 or overWrite != 1):
            print("Acceptable values are 1/0. Please retry")
            return
        self.overWrite = overWrite
        print("Data is not overwrite protected")

    def setTargetVarName(self,targetVarName):
        if self.DFCreated == 0:
            print("Please upload input dataset before setting target")
            return
        if not any(targetVarName in dUmmy for dUmmy in self.varListInpDF):
            print("Target variable: %s not found in input dataset" %targetVarName)
            return
        self.targetVarName = targetVarName
        self.inpSanityCheck = 1
        print("Target variable successfully set")
#                if self.inpDF[targetVarName].dtype == np.float64 or
#                    self.inpDF[targetVarName].dtype == np.int64:
#                        self.targetVarName = targetVarName
#                else:
#                    #need to code here
#                    self.inpDF = self.inpDF.rename(columns{targetVarName:"oldTargetVarName"})
#                    self.inpDF[targetVarName] = pandas.get_dummies(test["oldTargetVarName"], drop_first=True)
#                self.inpSanityCheck = 1
        self.uniqueTgtCount = self.inpDF.groupby(targetVarName)[targetVarName].nunique().sum()

    def setPlotOneTgtClass(self,plotOneTgtClass):
        if not self.inpSanityCheck:
            print("Please set input dataframe and target variable first")
            return
        if self.uniqueTgtCount == 2 and plotOneTgtClass:
            self.plotOneTgtClass = plotOneTgtClass
            print("Plotting only one target class")
        else:
            self.plotOneTgtClass = 0
            print("Plotting all target classes")


    #Exception handling and sanity checks on the dataframe created needs to be created

    #Constructor for uploading CSV. By default header is present, set to "N" for no header.
    def uploadCSV(self,CSVPath, headerPresent = "Y"):
        #Check overwrite protect on existing dataset
        if self.DFCreated and not self.overWrite:
            print("Upload failed. Overwrite protected")
            return
        if headerPresent == "Y":
            self.inpDF = pandas.read_csv(CSVPath)
        else:
            self.inpDF = pandas.read_csv(CSVPath,
                                     header=None)
        #Create a list of column names of the uploaded dataset
        self.varListInpDF = list(self.inpDF.columns.values)
        self.DFCreated = 1
        print("CSV upload successful")
        self._stripDF()
        self.overWrite = 0

    #Constructor for uploading an existing pandas dataframe
    def uploadDF(self,inpDF):
        #Check overwrite protect on existing dataset
        if self.DFCreated and not self.overWrite:
            print("Upload failed. Overwrite protected")
            return
        self.inpDF = inpDF
        #Create a list of column names of the uploaded dataset
        self.varListInpDF = list(self.inpDF.columns.values)
        self.DFCreated = 1
        print("Data upload successful")
        self._stripDF()
        self.overWrite = 0

    def _stripDF(self):
        #Strip column names to remove whitespace
        self.inpDF.columns = [w.strip() for w in self.inpDF.columns.values]
        #Strip whitespace in column values
        self.inpDF = self.inpDF.applymap(lambda x:x.strip() if type(x) is str else x)


    #Preserve continuous variables distribution - 1/0. Default - 0
    def setPreserveContDist(self,preserveContDist):
        if type(preserveContDist) is not int and (preserveContDist != 0 or preserveContDist != 1):
            print("Acceptable values are 1/0. Please retry")
            return
        self.preserveContDist = preserveContDist
        if self.preserveContDist:
            print("Distribution of continuous variable will be preserved")
        else:
            print("Continuous variable will be split into %s equal parts"%self.numBins)

    #Set number of bins for splitting continuous variables
    def setNumBins(self,numBins):
        if type(numBins) is not int:
            print("Number of bins is not integer")
            return
        if numBins == 0:
            print("Number of bins entered 0. Setting default value of 10")
            return
        self.numBins = numBins
        print("Number of bins changed to %s"%self.numBins)


    #Set how much portion of continuous data outlier to be cut before bucketing it.
    #This avoids outliers influencing the buckets.
    def setQuantileCut(self,quantileCut):
        if (type(quantileCut) is not int) and quantileCut <= 1:
            self.quantileCut = min(quantileCut,1-quantileCut)
        else:
            print("Quantile cut seems to be invalid. Reverting to default value of 0.05")

    #Set maximum categories till when bivariate shall be safely performed. Set to 30 by default
    def setMaxCatCutoff(self,maxCat):
        if type(maxCat) is not float and type(maxCat) is not int:
            print("Argument shall not be character")
            return
        if maxCat < 2:
            print("Maximum categories shall not be less than 2")
            return
        self.maxCatCutoff = maxCat
        print("Maxmimum category cut off changed")

    #Treat missing in bivar. Default - 0(No)
    def setBivarKeepMissing(self,bivarKeepMissing):
        if type(bivarKeepMissing) is not int:
            print("Argument shall not be character. Reverting to default:0")
            return
        if bivarKeepMissing != 0 and bivarKeepMissing != 1:
            print('Argument shall be 0/1 only. Reverting to default:0')
            return
        self.bivarKeepMissing = bivarKeepMissing
        print("Missing values will be retained in bivariate")

    #Create bivariate
    #If varlist is empty, bivariate will be created for all independent var
    #If not empty, varlist needs to be a list of input variables.
    def createBivar(self,varList=[]):
        #Check if target variable set
        if self.targetVarName == "":
            print("Target variable name not set. Please set target variable name")
            return

        #Display working directory and get confirmation from user
        #Would help if saving image or pdf of plots
        #Prompt only once
        if not self.cwdConfimation:
            print("Current working directory is %s"%os.getcwd())
            userInp = input("Do you want to continue [Y/N]:")
            if userInp != 'Y':
                print("Please use method setCurrentDir to set intended directory")
                return
            self.cwdConfimation = 1

        #Check if bivariate needs to be performed for all variables.
        #This may be time consuming and could be avoided at this step
        #Check also if it is a list, otherwise stop exec
        if varList == []:
            userInp = input("Process bivariate for all variables. Do you want to continue [Y/N]:")
            if userInp != 'Y':
                return
            varList = self.varListInpDF
        else:
            if type(varList) is not list:
                print(varList)
                print(type(varList))
                print('Entered arg is not list, stopping execution')
                return

        #For each variable in varList, create bivariate
        for varName in varList:
            if varName != self.targetVarName:
                varDType = self.inpDF[varName].dtype
                uniqueVarCount = self.inpDF.groupby(varName)[varName].nunique().sum()
                #If variable is categorical and more than specified number of unique categories, do not perform bivariate
                if uniqueVarCount < self.maxCatCutoff or (varDType != np.float64 and varDType != np.int64):
                    if uniqueVarCount < self.maxCatCutoff:
                        #bivarTab is for printing. It has total along with categories and counts
                        #bivarPlotLine is for plotting event percentage line
                        if self.bivarKeepMissing:
                            bivarTab = pandas.crosstab(self.inpDF[varName].fillna('Missing'),self.inpDF[self.targetVarName],margins=True)
                            bivarPlotLine = pandas.crosstab(self.inpDF[varName].fillna('Missing'),self.inpDF[self.targetVarName],normalize='index')
                        else:
                            bivarTab = pandas.crosstab(self.inpDF[varName],self.inpDF[self.targetVarName],margins=True)
                            bivarPlotLine = pandas.crosstab(self.inpDF[varName],self.inpDF[self.targetVarName],normalize='index')
                        #For two categories, there is option of plotting only one class percentages
                        if self.plotOneTgtClass:
                            del bivarPlotLine[bivarPlotLine.columns[1]]
                        #bivarPlotBar is for plotting categorical distribution percentages as bar
                        if self.bivarKeepMissing:
                            bivarPlotBar = pandas.crosstab(index=self.inpDF[varName].fillna('Missing'),columns=[varName],normalize='columns')
                        else:
                            bivarPlotBar = pandas.crosstab(index=self.inpDF[varName],columns=[varName],normalize='columns')
                        #Plot count percentages in primary axis
                        ax=bivarPlotBar.plot(kind="bar",color='Orange')
                        #Plot event count in secondary axis
                        #Unable to set secy axis limit in this step
                        bivarPlotLine.plot(secondary_y=True,ax=ax,ylim=[0,1],mark_right = False)
                        #Print tabular form of bivariate
                        print(bivarTab)
                        #Need to export to pdf or something
                    else:
                        print("Too many categories in variable %s. Not processing bivariate"%varName)
                else:
                    #All of this to capture outliers since specified quantile would be cut before binning
                    minVal = self.inpDF[varName].min()
                    maxVal = self.inpDF[varName].max()
                    if minVal < 0:
                        minValToSet = minVal*100
                    else:
                        minValToSet = minVal - (minVal*100)
                    if maxVal < 0:
                        maxValToSet = maxVal - (maxVal*100)
                    else:
                        maxValToSet = maxVal*100
                    bins = []
                    binLabel = []
                    #This preserves original distribution of the continuous variable
                    if self.preserveContDist:
                        minBin = self.inpDF[varName].quantile(self.quantileCut)
                        maxBin = self.inpDF[varName].quantile(1-self.quantileCut)
                        #Just a check that if there is no variability in data, do not process
                        if minBin != maxBin:
                            prevLoopCtr = minValToSet
                            #Split data into equally sized bins preserving original distribution
                            for loopCtr in np.arange(minBin,maxBin,(maxBin-minBin)/self.numBins):
                                if loopCtr == minBin:
                                    bins.append(minValToSet)
                                    initLoop = 1
                                else:
                                    bins.append(loopCtr)
                                    if initLoop:
                                        binLabel.append(".min"+' -- '+str(loopCtr))
                                        initLoop = 0
                                    else:
                                        binLabel.append(str(prevLoopCtr)+' -- '+str(loopCtr))
                                    prevLoopCtr = loopCtr
                            bins.append(maxValToSet)
                            binLabel.append(str(prevLoopCtr)+' -- '+"max")
                        else:
                            print("Variable %s seems not to have a lot of variability"%varName)
                    else:
                        #Split data into equally sized groups - all bins will have equal population
                        varRanks = pandas.qcut(self.inpDF[varName],self.numBins,labels=False,duplicates='drop')
                        varNameRank = varName + "_ranks"
                        #Temporarily set a column with ranks
                        self.inpDF[varNameRank] = varRanks
                        bins.append(minValToSet)
                        prevAppendVal = minValToSet
                        initLoop = 1
                        for appendVal in self.inpDF[[varName,varNameRank]].groupby(varNameRank).min()[varName]:
                            bins.append(appendVal)
                            if initLoop:
                                binLabel.append(".min"+' -- '+str(appendVal))
                                initLoop = 0
                            else:
                                binLabel.append(str(prevAppendVal)+' -- '+str(appendVal))
                            prevAppendVal = appendVal

                        bins.append(maxValToSet)
                        binLabel.append(str(prevAppendVal)+' -- '+"max")
                        del self.inpDF[varNameRank]

                    #Set bins to the bin min max range. This categorical variable will be used to plot bivariate
                    binValues = pandas.cut(self.inpDF[varName], bins, labels=binLabel)
                    varNameBins = varName + "_bin"
                    self.inpDF[varNameBins] = binValues
                    if self.bivarKeepMissing:
                        self.inpDF[varNameBins] = self.inpDF[varNameBins].cat.add_categories('Missing')
                        bivarTab = pandas.crosstab(self.inpDF[varNameBins].fillna('Missing'),self.inpDF[self.targetVarName],margins=True)
                        bivarPlotLine = pandas.crosstab(self.inpDF[varNameBins].fillna('Missing'),self.inpDF[self.targetVarName],normalize='index')
                    else:
                        bivarTab = pandas.crosstab(self.inpDF[varNameBins],self.inpDF[self.targetVarName],margins=True)
                        bivarPlotLine = pandas.crosstab(self.inpDF[varNameBins],self.inpDF[self.targetVarName],normalize='index')
                    #Option of plotting only one target class, only available for two class target
                    if self.plotOneTgtClass:
                        del bivarPlotLine[bivarPlotLine.columns[1]]
                    bivarPlotBar = pandas.crosstab(index=self.inpDF[varNameBins],columns=[varNameBins],normalize='columns')
                    ax=bivarPlotBar.plot(kind="bar",color='Orange')
                    bivarPlotLine.plot(secondary_y=True,ax=ax,ylim=[0,1],mark_right = False,rot=90)
                    print(bivarTab)
                    print(bivarPlotBar)
                    print(bivarPlotLine)

    #Generate univariate distribution
    #If varlist is empty, bivariate will be created for all independent var
    #If not empty, varlist needs to be a list of input variables.
    def createUnivar(self,varList,reqStatsList = []):
        #Display working directory and get confirmation from user
        #Would help if saving image or pdf of plots
        #Prompt only once
        univarSingleVar = 0
        univarReqStats = 0
        if not self.cwdConfimation:
            print("Current working directory is %s"%os.getcwd())
            userInp = input("Do you want to continue [Y/N]:")
            if userInp != 'Y':
                print("Please use method setCurrentDir to set intended directory")
                return
            self.cwdConfimation = 1

        #Check if univariate needs to be performed for all variables.
        #This may be time consuming and could be avoided at this step
        #Check also if it is a list, otherwise stop exec
        if varList == []:
            userInp = input("Process univariate for all variables. Do you want to continue [Y/N]:")
            if userInp != 'Y':
                return
            varList = self.varListInpDF
        else:
            if type(varList) is not list:
                print(varList)
                print(type(varList))
                print('Entered arg is not list, stopping execution')
                return
            univarSingleVar = 1

        #Check if all univariate stats needs to be performed or specified.
        #Check also if it is a list, otherwise stop exec
        if reqStatsList != []:
            if type(reqStatsList) is not list:
                print('Entered arg is not list, stopping execution')
                return

            if any("Missing_Stats" in dUmmy for dUmmy in reqStatsList):
                self._createMissingStats(varList);
                univarReqStats = 1

            if any("Univariate_Stats" in dUmmy for dUmmy in reqStatsList):
                self._createUnivarStats(varList);
                univarReqStats = 1

            if any("Univarite_Plot" in dUmmy for dUmmy in reqStatsList):
                #createUnivarPlot(varList);
                univarReqStats = 1
        else:
            userInp = input("Processing all univariate statistics and plots. Continue Y/N:")
            if userInp != 'Y' and userInp != 'y':
                print("Stopping Execution")
                return
            self._createMissingStats(varList)
            self._createUnivarStats(varList)
            #_createUnivarPlot(varList)

            #Code all necessary subs

    #Only called inside the class. This ensures all sanity check done and no untoward error is thrown.
    #Create missing data statistics for the requested list of variables
    def _createMissingStats(self,varList):
        #For each variable in varList, create missing variable report
        #Get count of rows for percentage missing
        DFRowCount = self.inpDF.shape[0]
        #variable index for printing
        varIndex = 1
        #Create an empty report dataframe
        missingReport = pandas.DataFrame()

        for varName in varList:
            #Get number of missing values for the variable
            numMissing = np.sum(pandas.isnull(self.inpDF[varName]))
            numNotMissing = DFRowCount - numMissing
            #Percentage missing
            percentMissing = str(round(numMissing*100/DFRowCount,2)) + '%'
            percentMissingPlot = round(numMissing*100/DFRowCount,2)
            #percentNotMissing = str(round(numNotMissing*100/DFRowCount,2)) + '%'
            percentNotMissingPlot = round(numNotMissing*100/DFRowCount,2)
            #Append each variables stats with proper index
            missingReport = missingReport.append(
                    pandas.DataFrame({'Variable Name':varName,
                                      'Missing Count': numMissing,
                                      'Non-missing Records Count':numNotMissing,
                                      'Missing Percentage': percentMissing
                                      },columns = ['Variable Name',
                'Missing Count',
                'Non-missing Records Count',
                'Missing Percentage'
                ],index=[varIndex]))
            varIndex = varIndex + 1
            missingPlot = pandas.DataFrame({'Variable Name':'Missing',
                                            'Count':percentMissingPlot
                                            },columns=['Variable Name',
            'Count'],index=[1])
            missingPlot = missingPlot.append(pandas.DataFrame({'Variable Name':'Non-Missing',
                                                 'Count':percentNotMissingPlot
                                                 },columns=['Variable Name',
            'Count'],index=[2]))
            missingPlot.plot(x='Variable Name',y='Count',kind='bar')

        print('Total records in the input dataframe: %s' %DFRowCount)

        print("Missing data report for the requested columns: ")
        print(missingReport)

    def _createUnivarStats(self,varListInp):
        #Create an empty report dataframe
        univarReport = pandas.DataFrame()
        varList = list()

        #modeReport = self.inpDF[[varListInp]].astype('str').mode().fillna(value = '-').transpose()

        for varName in varListInp:
            if self.inpDF[varName].dtype != np.int64 and self.inpDF[varName].dtype != np.float64:
                print("Can't process descriptive statistics for categorical data: %s" %varName)
            else:
                varList.append(varName)

        varList = list(set(varList))
        if len(varList) == 0:
            print('No variables to perform descriptive statistics')
            return

        inpDFCut = self.inpDF[varList]

        #Get descriptive stats
        #Keep only one of the mode values
        #Mean
        modeDF = inpDFCut.mode().loc[[0]].astype('str').transpose()
        modeDF.columns = ['Mode']
        #Median
        meanDF = pandas.DataFrame(inpDFCut.mean().round(2).astype('str'))
        meanDF.columns = ['Mean']
        #Variance
        varianceDF = pandas.DataFrame(inpDFCut.var().round(2).astype('str'))
        varianceDF.columns = ['Variance']
        #Standard Deviation
        stdDF = pandas.DataFrame(inpDFCut.std().round(2).astype('str'))
        stdDF.columns = ['Standard Deviation']
        #Quantile
        quantileDF = inpDFCut.quantile([0,0.01,0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95,0.99,1]).transpose()
        quantileDF.columns = ['Min',
                              '1%',
                              '5%',
                              '10%',
                              '25% - Q1',
                              'Median',
                              '75% - Q3',
                              '90%',
                              '95%',
                              '99%',
                              'Max']
        #Inter Quartile Range
        quantileDF['IQR'] = quantileDF['75% - Q3'] - quantileDF['25% - Q1']
        #Range
        quantileDF['Range'] = quantileDF['Max'] - quantileDF['Min']
        #Transposed to keep variable names as row index. String conversion to prevent many decimal places for integer values
        quantileDF = quantileDF.round(2).transpose().astype('str').transpose()
        #Create the report dataframe
        univarReport = pandas.concat([meanDF,modeDF,varianceDF,stdDF,quantileDF],axis=1)
        print("Descriptive Statistics for the requested columns: ")
        print(univarReport)

    def removeMultiCollinearity(self):
        thresholdVIF = 5.0
        numColumns =
