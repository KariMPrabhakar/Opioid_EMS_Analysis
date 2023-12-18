# Opioid_EMS_Analysis - "Exploring the key factors influencing Narcan administration during opioid-related EMS calls in Tempe, Arizona. 

# Business Question - "What are the key factors associated with the administration of Narcan during opioid-related EMS calls, and how do these factors vary across different demographic groups, times of the day, or geographic locations?"


# Introduction

The opioid crisis remains a critical public health concern, with far-reaching implications for communities. This analysis aims to shed light on key factors associated with the administration of Narcan during opioid-related EMS calls for the data collected in Tempe, Arizona. Narcan, or naloxone, is a medication that can rapidly reverse opioid overdose and is a key component in the ongoing efforts to address the opioid crisis. By understanding these factors, we can enhance our response to opioid incidents and, ultimately, save lives.

# Graph 1: Temporal Distribution of Opioid Incidents - "Distribution of Opioid use Incidents Based on Time of Day" and "Narcan Given Incidents Based on Time of Day"

![Distribution of Opioid and Narcan use Incidents Based on Time of Day](https://github.com/KariMPrabhakar/Opioid_EMS_Analysis/blob/main/Narcan_Opioid_Time_of_Day.png).

This graph gives iinitial insights into the broader distribution of opioid-related incidents, illustrating the temporal patterns of opioid-related incidents and narcan administration throughout the day. By examining the frequency of incidents at different times, we can identify peak periods and potential trends in opioid and narcan-need emergencies. The analysis of the time distribution for EMS calls with opioid use incidents reveals that afternoon and evening are the most prevalent times. These findings suggest a potential pattern in the timing of opioid-related incidents, emphasizing the importance of resource allocation and emergency response planning during these periods. The analysis of Narcan given incidents based on time of day reveals that afternoon and evening exhibit the highest counts, suggesting potential peak periods for opioid-related emergencies. The provided data suggests an interesting observation: the number of Narcan administrations appears to be higher than the number of reported opioid use incidents in some years, suggesting that Narcan may have been administered in incidents where opioid use was reported as 'No' due to factors such as observed symptoms, uncertainties in self-reporting, or other clinical considerations. This could be due to various factors such as data completedness, narcan for non-opioid related incidents, training and awareness.


# Graph 2 & 3: Comparative Analysis - "Narcan Administration Across Years" and "Opioid Use Across Years"

![Narcan Administration Across Years and Opioid Use Across Years](https://github.com/KariMPrabhakar/Opioid_EMS_Analysis/blob/main/Narcan_Opioid_Across_Years.png)

The analysis of Narcan administration across years reveals a notable increase from 2017 to 2021, indicating a rising trend in opioid-related incidents during this period. However, in 2022, there is a noticeable drop, possibly due to limited data availability. It's essential to interpret the trends cautiously, considering data completeness for each year. The analysis of opioid use incidents over the years reveals a fluctuating pattern. Incidents rise from 2017 to 2019, decline in 2020, increase again in 2021 and show a notable drop thereafter but could be due to limited data. The analysis of opioid use incidents and Narcan administrations from 2017 to 2021 reveals varying trends, with opioid use incidents showing fluctuations and Narcan administrations exhibiting an overall increase. There is a generally positive correlation between opioid use incidents and Narcan administrations, suggesting that as opioid use incidents increase, so do Narcan administrations. However, an anomaly is observed in 2022, where both opioid use incidents and Narcan administrations show a decline, indicating a potential deviation from the established positive correlation. 



# Graph 4: "Distribution of Gender in Opioid Use 

![Distribution of Gender in Opioid Use](https://github.com/KariMPrabhakar/Opioid_EMS_Analysis/blob/main/Distribution_of_Gender_in_Opioid_Use.png)

The bar plot illustrates the distribution of gender within the subset of opioid-related incidents where opioid use is recorded as 'Yes.' The visual representation showcases the count of male,
and female responses only, eliminating the unknown responses. This focused analysis provides insights into the gender distribution specifically among cases where opioids were confirmed to be used. 


# Narcan Related Demographics 


## Graph 5: "Narcan Given Incidents Across Age Groups"

![Narcan Given Incidents Across Age Groups](https://github.com/KariMPrabhakar/Opioid_EMS_Analysis/blob/main/Narcan_Given_Incidents_Across_Age_Groups.png)

After exploring the relationship between age and Narcan given incidents using age bins, it is evident that age groups 19-30 and 31-45 have the highest counts of "Yes" responses to Narcan given across the age groups.


## Graph 6: Narcan Given Incidents Across Specific Populations

![Narcan Given Incidents Across Specific Populations](https://github.com/KariMPrabhakar/Opioid_EMS_Analysis/blob/main/Narcan_Given_Incidents_Across_Specific_Populations.png)

Based on the analysis of Narcan given incidents across specific populations, it is observed that the homeless population has the highest count.



# Heat map of Narcan given incidents in Tempe, Arizona

![Heatmap of Narcan Given Incidents](https://github.com/KariMPrabhakar/Opioid_EMS_Analysis/blob/main/Tempe_Heat_Map.docx)

Geographically, Marina Heights, Cameron Way, Hermosa Drive, and Veteran's Way College show higher narcan administration incidences in the Tempe, Arizona area. Consideration of location-specific trends could further enrich the understanding of the opioid crisis and narcan administration needs.



## Predictive Model: Decision Tree Classifier

To complement the insights gained from the exploratory analysis, a Decision Tree Classifier model was trained to predict whether Narcan was administered during opioid-related EMS calls. The model achieved a precision of 66%, correctly predicting Narcan administrations for 66% of the cases in the test set, a prediction that can be improved on. The predictive model incorporates features such as age, weekday, opioid use, and specific populations to make these predictions.


## Model Training Code

```python
# I imported the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Loaded my dataset
EMS_df = pd.read_csv('/content/drive/MyDrive/Opioid_EMS_Calls.csv')

# Convert 'Narcan_Given' to a binary format
EMS_df['Narcan_Given'] = EMS_df['Narcan_Given'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop rows with missing values 
data = EMS_df[['Age', 'Weekday', 'Opioid_Use', 'Spec_Pop', 'Narcan_Given']].dropna()

# Convert age ranges to midpoint because ages were categorical ranges
def age_range_to_midpoint(age_range):
    if age_range == 'Unknown':
        return None
    start, end = map(int, age_range.split(' to '))
    return (start + end) / 2

data['Age'] = data['Age'].apply(age_range_to_midpoint)

# Drop rows with missing values after age conversion
data = data.dropna()

# Encode categorical variables
label_encoder = LabelEncoder()
data['Weekday'] = label_encoder.fit_transform(data['Weekday'])
data['Opioid_Use'] = label_encoder.fit_transform(data['Opioid_Use'])
data['Spec_Pop'] = label_encoder.fit_transform(data['Spec_Pop'])

# Separate features and target variable
X = data[['Age', 'Weekday', 'Opioid_Use', 'Spec_Pop']]
y = data['Narcan_Given']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Next we code to build and train the model we will be using
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Saving the model to a file we can use for later training
joblib.dump(model, '/content/drive/MyDrive/Opioid_EMS_Calls_Proj/narcan_prediction_model.joblib')


# For new training 
new_data = pd.DataFrame(...)  # Here, we can replace ... with the actual data
predictions = model.predict(new_data)
print(predictions)



## Conclusion

This analysis has explored key factors associated with Narcan administration during opioid-related EMS calls. The insights gained contribute to a better understanding of how Narcan can be strategically employed to address opioid emergencies and, in turn, enhance our overall response to the opioid crisis.

By uncovering trends and demographic details related to the data collected, opioid-related EMS calls and the administration of Narcan has shown key insights into factors associated with these incidents. Demographically, there is a notable concentration of Narcan administrations among individuals aged 19-30 and 31-45, with a higher prevalence among the homeless population. Temporally, the afternoon and evening exhibit heightened occurrences of opioid-related emergencies, emphasizing the need for strategic resource allocation during these periods. Additionally, a temporal analysis across years suggests a rising trend in opioid-related incidents from 2017 to 2021, with a subsequent decline in 2022. Geographically, Marina Heights, Cameron Way, Hermosa Drive, and Veteran's Way College show higher narcan administration incidences in the Tempe, Arizona area. Consideration of location-specific trends could further enrich the understanding of the opioid crisis. The Decision Tree Classifier model, incorporating demographic and temporal features, contributes predictive capabilities with a 66% precision in determining Narcan administrations. These findings collectively illuminate the multifaceted nature of opioid-related emergencies, offering valuable insights for targeted interventions, resource optimization, and comprehensive emergency response planning across diverse demographic groups and temporal contexts.


