# **Mass Mobilization**

## Table of Contents

1. [Executive Summary](#1.Executive-Summary)
2. [Data Collection](#Data-Collection)
3. [Data Cleaning & Pre-Processing](#3.-Data-Cleaning-&-Pre-Processing)
4. [EDA](#4.-EDA)
5. [Modeling](#5.-Modeling)
6. [Evaluation & Analysis](#6.-Evaluation-&-Analysis)
7. [Conclusion & Recommendations](#7.-Conclusion-&-Recommendations)
8. [Next Steps & Future work](#8.-Next-Steps-&-Future-work)
9. [Citations](#9.-Citations)
10. [Acknowledgements](#10.-Acknowledgements)


## 1. Executive Summary

In light of the mass mobilization events over the past year, protests are basically commonplace in modern day America. More importantly, the way the US government responds is under increasing scrutiny. For this reason, the US government ought to have better tools for understanding and preparing for protests and demonstrations.

The Mass Mobilization Project is a site that hosts data about citizen movements against governments. The original intent of the MM study was to inform foreign policy and understand the impact of mass mobilizations outside the United States. The data contains around 17,000 protests where 50 or more individually publically protested against government. More information can be found at the website: massmobilization.github.io.

Our goal is to prepare the data for machine learning modeling and predict a government's response. In the end, we hope that given a good product that makes reasonably accurate predictions, we can yield meaningful insights from the data, infer relationships between public protests and governments responses, and add context to the ongoing conversation of protests in America.

## 2. Data Collection




## 3. Data Cleaning & Pre-Processing

The researchers at the Mass Mobilization project had scraped the protest data from news publication, so the resulting dataset revealed a high degree of variability and ambiguity in several of the features. The original dataset had over 17,000 protests included, so each one was a row in the DataFrame. In order to get the data ready for EDA and subsequent modeling, our team performed the following cleaning and pre-processing steps:

+ Dropped rows where `protest` column was equal to 0, as the project methodology gave no explanation for where the 0/1 distinction.
+ Bucketed `participantsidentity` into the following categories, which we then encoded as binary variables:
  - Civil/human rights groups
  - Ethnic groups
  - Local residents
  - Pensioners/retirees
  - Prisoners
  - Protesters (generic)
  - Religious groups
  - Soldiers/veterans
  - Students/youth
  - Families of victims
  - Women
  - Workers/unions
+ Encoded the following features as to create binary features for modeling:
  - country
  - region
+ `participants` and `participants_category` columns both gave information about the number of protesters involved and were both partially filled-in. We combined these into one column that had the range of protester numbers for every protest. We then encoded this new column so that we had seven new columns that each indicated whether a given protest had protester numbers recorded to be in that range:
  - Number of participants <50
  - 50 - 99 participants
  - 100 - 999 participants
  - 1,000 - 4,999 participants
  - 5,000 - 9,999 participants
  - 10,000 - 100,000 participants
  - Number of participants >100,000
+ Turned the four columns called `protesterdemand` into seven new binary-encoded columns, one for each protest motivation:
  - Labor & wage disputes
  - Land & farming conflicts
  - Police brutality
  - Political behavior or processing
  - Tax increases / tax policy
  - Removal of politician(s)
  - Social restrictions
+ Turned the seven columns called `stateresponse` into seven new binary-encoded columns, one for each of the state responses:
  - accomodation
  - arrests
  - beatings
  - crowd dispersal
  - ignore
  - killings
  - shootings

At the end of the data cleaning process, our new data landscape looked like this:

**Columns included in feature set:**

|Feature|Type|Description|
|---|---|---|
|protestnumber|int|Number of protests that year in specific country|
|protesterviolence|int(binary)|Indicates whether protester enacted violence|
|pop_total|float|Population of country where protest takes place|
|pop_density|float|Population density of country where protest takes place|
|prosperity_2020|float|Prosperity index of country the year protest takes place|
|region_Africa|int (binary)|Country is in Africa|
|region_Asia|int (binary)|Country is in Asia|
|region_Central America|int  (binary)|Country is in Central America|
|region_Europe|int (binary)|Country is in Europe|
|region_MENA|int (binary)|Country is in the Middle East/North Africa|
|region_North America|int (binary)|Country is in North America|
|region_Oceania|int (binary)|Country is in Oceania|
|region_South America|int (binary)|Country is in South America|
|protest_size_category_Less than 50|int (binary)|Number of participants <50|
|protest_size_category_50-99|int (binary)|50 - 99 participants|
|protest_size_category_100-999|int (binary)|100 - 999 participants|
|protest_size_category_1,000-4,999|int (binary)|1,000 - 4,999 participants|
|protest_size_category_5,000-9,999|int (binary)|5,000 - 9,999 participants|
|protest_size_category_10,000-100,000|int (binary)|10,000 - 100,000 participants|
|protest_size_category_Over 100,000|int (binary)|Number of participants >100,000|
|protester_id_type_civil_human_rights|int (binary)|Indicates participants are civil/human rights group|  
|protester_id_type_ethnic_group|int (binary)|Indicates participants are ethnic group|
|protester_id_type_locals_residents|int(binary)|Indicates participants are local residents|
|protester_id_type_pensioners_retirees|int (binary)|Indicates participants are pensioner/retirees|
|protester_id_type_prisoners|int (binary)|Indicates participants are prisoners|
|protester_id_type_protestors_generic|int (binary)|Indicates participants are generic group|
|protester_id_type_religious_group|int (binary)|Indicates participants are from religious group|    
|protester_id_type_soldiers_veterans|int (binary)|Indicates participants are soldiers/veterans|
|protester_id_type_students_youth|int (binary)|Indicates participants are students/youth|
|protester_id_type_victims_families|int (binary)|Indicates participants are families of victims|
|protester_id_type_women|int (binary)|Indicates participants are primarily groups of women|
|protester_id_type_workers_unions|int (binary)|Indicates participants are union members or workers|
|labor_wage_dispute|int( binary)|Protest motivation is labor & wage disputes|
|land_farm_issue|int (binary)|Protest motivation is land and farming conflict|
|police_brutality|int (binary)|Protest motivation is police brutality|
|political_behavior_process|int (binary)|Protest motivation is political behavior or process|
|price increases_tax_policy|int (binary)|Protest motivation is tax increase/tax policy|
|removal_of_politician|int (binary)|Protest motivation is removal of politician(s)|
|social_restrictions|int (binary)|Protest motivation associated with social restrictions|

**Other columns in original dataset that were not included in our final model:**

|Feature|Type|Description|
|---|---|---|
|id|int|Unique protest id number|
|country|object|Country of protest|
|ccode|int|Unique code for country|
|location|object|Description of where protest takes place within country|
|sources|object|Media outlets that reported on protest|
|notes|object|Description of protest|
|participants_number|int|Estimated number of participants|
|start_date|object|Recorded start date of protest|
|end_date|object|Recorded end date of protest|

**Columns included as target variables:**

|Target|Type|Description|
|---|---|---|
|accomodation|int (binary)|State response includes accommodation of protester demands|
|arrests|int (binary)|State response includes arrests of protesters|
|beatings|int (binary)|State response includes beating protesters|
|crowddispersal|int (binary)|State response includes crowd dispersal|
|ignore|int (binary)|State ignores protesters and protest demands|
|killings|int (binary)|State response includes killing protesters|
|shootings|int( binary)|State response includes shooting protesters|
|violent_response|int (binary)|State response includes one or more violent response (beatings, shootings, killings)|


## 4. EDA



## 5. Modeling

At first glance, we were presented with a multi-label classification problem i.e. a government can have multiple responses to the same protest. For these reasons, we initially built a couple of models: a Neural Network and a sklearn Multilabel Classifier using bagging and random forests ensemble methods. Our neural network model was promising given the first iteration, but we quickly realized we could not interpret any meaningful insights from it. Our Multilabel Classifier did not perform well.

For the second modeling effort, we built seven different logistic regressions for each of our target variables, effectively running a binary classification on each class. The results for these models varied widely and predictions were not reliable for classses that were less frequent in the data. This is a pitfall for having massively imbalanced data. More importantly to our interests, the violent responses were massively underepresented.

The third modeling effort was a slight spinoff from our second modeling effort. First, we ran a binary logistic regression on whether or not the state government ignored a protest. After running this model, we filtered our data to exclude points where the government did not ignore a protest (i.e. which can be interpeted as when the government did respond). We then ran six different binary logistic regressions for each of the remaining six variables. Treating the ignore logistic regression separately reduced imbalances in our data and improved our scores across the board, especially for our columns of interest: the violent responses.

Given a decent product, we ran a gridsearch on several hyperparameters with an emphasis on the 'class_weight' of our logistic regression, which effectively penalizes wrong predictions for the class at hand based on a ratio default is 1:1).

We even went a step further and grouped all three of the violent responses--beatings, killiings, and shootings--into one 'violent_response' column in order to reduce imbalances even more. However, this did not increase the scores of our model.

Please see the images folder in our repo for our final modeling technique's performance metrics, confusion matrices, and AUC plots.


## 6. Evaluation & Analysis



## 7. Conclusion

Given a model with reasonable scores, we were most interested in highlighting relationships in the data when the government either ignores protests or violently responds to them. Conclusions for the ignore target were relatively straight forward and expected. For instance, governments were more likely to ignore protests in countries with a higher prosperity index score. An interesting nuace in the findings is governments were less likely to ignore small protests and more likely to ignore larger protests. Intuitively, perhaps larger protests are more organized and anticipated whereas smaller protests are unpredictable, which is more likely to raise uncertainty for the government at hand.

As for violent responses, all three target variables (killings, shootings, and beatings) had similar important features. Obviously, if protesters were violent, the goverment responded violently. Europe was an important feature when the government is less likely to respond violently for two of the categories, and Asia was an important feature for violent responses of two of the categories as well.

All in all, it was a generally mixed bag as these relationships over a few dozen features is complex, but we feel good about our model and it's scores and feel relatively confident about any inferences our model reveals.


## 8. Next Steps & Future work

Given the nature the data and discourse, our goal for this project is to model state responses to protests in order to inference relationships between the data and state responses for further exploration. We believe there is significant work to be finished. Some areas include focusing on gathering more information and more effective non-categorical information for the variables our model deems as predictive for violent responses.

We also believe it is prudent to perform a similar analysis to ours with respect to individual countries. If that is too granular, then by continent.

In terms of the modeling, we believe our model can be improved and it's worth considering a couple of techniques. To balance the data more sufficiently, bootstrapping techniques might be applied and improve the classification metrics of our models. It's also possible that we could implement unsupervised clustering techniques to group variables and perform analysis on those clusters. Perhaps, there are inferences that can be observed that were not observed with our models.

Ultimately, gathering more data for the sake of creating more balanced classes would improve the predictive power of our models and the interpretability of them. With this in mind, we also believe it is prudent and absolutely necessary to gather the same data about US protests so that we can learn about mass mobilization at home.


## 9. Citations
The Mass Mobilization Project ([*source*](https://massmobilization.github.io/about.html)).


## 10. Acknowledgements

All of our data and inspiration for our project is from "The Mass Mobilization Project" conducted by Professor David H. Clark from Bighamton Universty and Professor Patrick M. Regan from the University of Notre Dame.

Thank you for your work in retrieving the data and providing a site for further exploration.



Thank you again.
