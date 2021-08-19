# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 17:04:09 2021
Model for currency forecasting using time series + Wavelets + SVR
@author: ksgordillo

"""

#Standard libraries
import numpy as np
import pandas as pd
import copy
#Libraries for SVM implementation
from sklearn import svm
#from sklearn.metrics import mean_squared_error
from numpy.lib.stride_tricks import sliding_window_view
#Library for Wavelets transformations, the code was taken from: https://github.com/pistonly/modwtpy/blob/master/modwt.py
from lib.modwt import modwt, imodwt


class WaveletSvrForecaster:
        
    __originalDates = None
    __originalPrices = pd.Series([])
    
    __newDates = None
    __newPrices = None
    
    __svrPredictors = []
    
    __trainModel = True
    
    '''
    - file_Path: File Path which contains the data
    - Delimiter: delimiter to split the data
    - dateColName: Date column name
    - closingPColName: Closing prices column name
    
        returns: arrays with dates and closing prices
    '''
    def __readData(self, file_Path, dateColName, closingPColName, delimiter):
        #Reading the data from csv file
        #We use 'parse_dates' to convert the date string into objects we can work with
        prices = pd.read_csv(file_Path, delimiter, header=0, encoding='utf-8', parse_dates=[dateColName])
        # defining variables
        dates = prices[dateColName].copy()
        closing_prices = prices[closingPColName].copy()
    
        return dates, closing_prices

    '''
    Using modwt with 'sym4' wavelet and 5 levels 
    (4 detail coffiecients (dC) and 1 approximation coefficient (aC))
    
    - _data: time series (TS) to decompose
    - type: type of wavelet to use to decompose the TS, default wavelet: sym4
    - _level: number of levels to decompose the TS
    '''
    def __applyModwt(self, _data, type='sym4', _level=3):
        #Getting wavelet coefficients
        _coeff = modwt(_data, type, _level)
        return _coeff
    
    '''
    Calling wavelet function for closing prices
    - series_values: values to decompose
    '''
    def __getCoeffFromSeries(self, series_values, waveletLevel):
        #calling function defined previously
        coeff = self.__applyModwt(series_values,type='sym4',_level=waveletLevel)
        return coeff
    
    '''
    Function to train the model using Radial basis function Kernel (rbf)
    '''
    def __trainModelCoeff(self, X, Y):
        #Values needed to fully reconstruct the time series
        svr = svm.SVR(kernel ='rbf', C=1e3, gamma=0.1)
        svr.fit(X, Y)
        
        return svr
    
    def __predictNewCoeff(self, coeff, prediction_days, past_days):
        new_coeff = []
        for i in range(len(coeff)):
            
            X, Y = self.__slideWindow(coeff[i], past_days)       
            
            if self.__trainModel:
                self.__svrPredictors.append(self.__trainModelCoeff(X, Y))
            
            svr = self.__svrPredictors[i]
            
            predictCoeff = self.__evaluateModel(svr, X, Y, prediction_days, past_days)
            
            newCoeff_concat = np.concatenate((coeff[i][:-1], predictCoeff))
            
            new_coeff.append(newCoeff_concat)
            
        return new_coeff
    
    '''
    Implementing slide window
    '''
    def __slideWindow(self, series, window_lenght = 2):

        _X, _Y = [], []
        #Auxiliary variable to store the sliding window combinations. We sum up +1 as we are taking the last values of Aux_window
        #as the output values of our time series
        aux_Window =  sliding_window_view(series, window_lenght+1)
        # Taking first 'window_lenght' values as the input (X) and the last value (window_lenght+1) as the output (Y)
        for i in range(len(aux_Window)):
            _Y.append(aux_Window[i][-1])
            _X.append(aux_Window[i][:-1])
        
        return _X, _Y
    
    '''
    Method used to make slide window and prediction
    '''
    def __evaluateModel(self, svr, X, Y, prediction_days, past_days):
        X_ = []
        Y_ = []
        Y_.append(np.array(Y)[-1])
        X_.append(X[-1])
        for i in range(prediction_days):
            Y_array = np.array([Y_[-1]])
            X_array = np.array(X_[-1][-past_days+1:])
            X_Y_concat = np.array([np.concatenate((X_array,Y_array))])
            X_ = np.concatenate(( X_, X_Y_concat ))
            p_value = svr.predict(X_[-1].reshape(1, -1))
            Y_ = np.concatenate(( Y_,  p_value))
        return Y_
    
    '''
    Aux method to calculate the new dates of the prediction
    '''
    def __addDayToDates(self, prediction_days):
        self.__newDates = copy.deepcopy(self.__originalDates)
        lastDate = np.array(self.__originalDates)[-1]
        for i in range (prediction_days+1):
            newDate = pd.to_datetime(lastDate) + pd.DateOffset(days=i)
            self.__newDates[len(self.__newDates)-1+i] = newDate
    
    '''
    Method to load the data, decompose the time series using wavelets and train the model using SVR
    
    In parameters: 
        
    - file_Path: File Path which contains the data
    - dateColName: Date column name
    - closingPColName: Closing prices column name
    - Delimiter: delimiter to split the data
    
    Returns:
    
    Trained SVR model with the incoming data
    
    '''
    def initializeModel(self, file_Path = '../Data/AUD-JPY-2003-2014-day.csv', dateColName = 'Date', 
                        closingPColName = 'Close', delimiter = ';'):
        
        #Loading data
        dates, closing_prices = self.__readData(file_Path, dateColName, closingPColName, delimiter)
        
        if closing_prices.equals(self.__originalPrices) != True:
            print('New data arrived, training again the model')
            self.__trainModel = True
        else:
            self.__trainModel = False
        
        #Storing dates and prices locally (in the object)
        self.__originalDates = dates
        self.__originalPrices = closing_prices
        
    
    
    '''
    Method to perform the prediction, it recieves the SVR model and re-constructed time series
    
    - past_days: days to take in consideration to perform the target prediction
    - daysToPredict: number of days to predict
    '''
    def makePrediction(self, daysToPredict = 7, past_days = 14):
        
        #Constant, wavelet level of decomposition
        waveletLevel = 4
        
        # Getting approximation coefficients and detail coefficients
        aCdC = self.__getCoeffFromSeries(self.__originalPrices, waveletLevel)
        
        
        # Getting new coefficients from the multi resolution analysis
        predictedCoeff = self.__predictNewCoeff(aCdC, daysToPredict, past_days)
        
        #Recombining detail and approximation coefficients after prediction
        predictedTimeSeries = imodwt(predictedCoeff,'sym4')
        
        
        #Getting just last values
        predicted_values = predictedTimeSeries[-daysToPredict:]

        self.__addDayToDates(daysToPredict)
        self.__newPrices = predictedTimeSeries
        
        return predicted_values, self.__newDates[-daysToPredict:]