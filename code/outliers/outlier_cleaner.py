#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    ### your code goes here
    for idx, val in np.ndenumerate(ages):
        error = residual_error(predictions[idx], net_worths[idx])
        cleaned_data.append((ages[idx], net_worths[idx],error))
        
    
    cleaned_data.sort(key=lambda row: row[2])
    size = len(cleaned_data)
    print("size before slicing %d", size)
    slice_len = int(0.9 * size)
    print("sliced array len %d", slice_len)
    cleaned_data = cleaned_data[:slice_len]
    return cleaned_data

def residual_error(prediction, net_worth):
    return ((prediction - net_worth)**2)