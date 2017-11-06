#!/usr/bin/python


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
    outliers = []
    data_Length = max(len(predictions),len(ages),len(net_worths))
    for a in range(0,data_Length):
        cleaned_data.append((ages[a],net_worths[a],net_worths[a]-predictions[a]))

    while len(cleaned_data) > (.9 * data_Length):
        cleaned_Data_Length = len(cleaned_data)
        error_Max = 0
        outlier_Row = 0
        for a in range(0,cleaned_Data_Length):
            if abs(cleaned_data[a][2]) > error_Max:
                error_Max = abs(cleaned_data[a][2])
                outlier_Row = a
        outliers.append(cleaned_data.pop(outlier_Row))

    return cleaned_data

