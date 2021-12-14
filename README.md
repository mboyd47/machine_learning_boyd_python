## Project Purpose
The Intensive Care Unit, or ICU as it is commonly known, of a hospital is where the most critically ill patients are sent to ensure they receive the constant care they need. By the time a patient enters the ICU, their health has often deteriorated so far to the point that the intense care they receive in the ICU is their only chance at survival. The goal of this project is to identify early indicators of patient mortality when they are admitted to the ICU. Patients admitted to the ICU are all at a high risk for mortality due to the severeness of their condition but the successful identification of indicators specific to ICU patients to help prevent mortality could give all those working in the ICU a forewarning of patients who may need even more intense care than is already given.

## Tools Used
I completed this project in Python using the following packages: Scikit Learn, Seaborn, and Matplotlib. 
- [Seaborn and Matplotlib](https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
) were used to create the correlation heatmap to investigate feature correlations, leveraged in [data_process.py](data_process.py).
- [Scikit Learn](https://scikit-learn.org/stable/) was used to split the data into testing and training sets, create the classifier models, and compute model assessment metrics, leveraged in [models.py](models.py). 


## Results
The classification methods created to predict hospital ICU outcome were logistic regression, support vector machine, and gradient boosting. Since the dataset is unbalanced and the value to be predicted was ICU deaths, recall score was used as the main method for assessing model performance in order to focus on minimizing the number of false negatives (patients predicted to live who actually died). Area under the ROC curve was used as a secondary evaluation metric.

The model that performed the best overall is the Support Vector Machine classifier. As shown in Table 1, this model had the highest recall score of 97% and the highest AUC of 79%.

### Table 1. Results of classification models
|Model	                |Recall Score|Area Under the ROC Curve|
|-----------------------|------------|------------------------|
|Logistic Regression    | 94.87%     | 77.25%                 |
|Support Vector Machine | 97.44%     | 79.50%                 |
|Gradient Boosting      | 79.48%     | 53.86%                 |
|Multilayer Perceptron  | 43.59%     | 72.51%                 |


Patient mortality is a subject that has been researched and studied by many people in the hopes of reducing death rates in hospitals everywhere. The classification models created in this project are just a glimpse at the possibility that this field holds. The winning model had a 97% recall rate, and thus very few false negatives, but only a 79% AUC. Improving this model or identifying an even better one could have a monumental impact on the healthcare system. As more data becomes available, it could be used to train models like these, improving them more and more over time, and could eventually help ICU workers all over the world.
