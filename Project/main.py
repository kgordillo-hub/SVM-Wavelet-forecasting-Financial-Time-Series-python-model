# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 17:18:07 2021

@author: ksgor
"""
from model import WaveletSvrForecaster


def main():
    
    predictor = WaveletSvrForecaster()
    #trained model and slide window combination of values
    while(True):
        print_menu(predictor)

def print_menu(predictor):
    
    menu_options = {
    1: 'Predict',
    2: 'Exit',
    }
    
    for key in menu_options.keys():
        print (key, '--', menu_options[key] )
        
    option = int(input('Enter your choice: ')) 
    if option == 1:
        print('Doing prediction...')
        
        predictor.initializeModel()
        
        #Perform prediction
        predictionValues, dates = predictor.makePrediction()
        
        print(predictionValues)
        print(dates)
        
        print('Finishing...')
    elif option == 2:
        print('Thanks message before exiting')
        exit()
    
   

if __name__ == "__main__":
    main()