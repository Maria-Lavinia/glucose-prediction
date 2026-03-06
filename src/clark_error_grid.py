from matplotlib import pyplot as plt
import pandas as pd


def get_clark_error_zone(predictedValues, actualValues):
    clark_error_zones = []
    
    for predicted, actual in zip(predictedValues, actualValues):
       
        if (actual <= 70 and predicted <= 70) or (predicted <= 1.2*actual and predicted >= 0.8*actual):
            clark_error_zones.append('A')   
        elif (actual <= 70 and predicted >= 180) or (actual >= 180 and predicted <= 70):
            clark_error_zones.append('E')
        elif(actual <= 70 and 70 <= predicted <= 180) or (actual >= 240 and 70 <= predicted <= 240):
               clark_error_zones.append('D')
        elif(70 <= actual <= 130 and predicted >= 180) or (130 <= actual <= 180 and predicted <= 70):
                clark_error_zones.append('C')     
        else:
                clark_error_zones.append('B')
    
    return clark_error_zones

def plot_clark_error_grid(zones, model_data_folder, test_patient):
    zone_counts = pd.Series(zones).value_counts()
    plt.figure(figsize=(8, 6))
    zone_counts.plot(kind='bar', color=['green', 'yellow', 'orange', 'red', 'black'])
    plt.xlabel('Clark Error Zone')
    plt.ylabel('Count')
    plt.title(f'Clark Error Zones for {test_patient}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{model_data_folder}/clark_error_zones_{test_patient}.png")
    plt.show()