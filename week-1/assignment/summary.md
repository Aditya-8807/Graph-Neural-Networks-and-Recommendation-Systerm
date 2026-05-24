# Metrics Summary
## Accuracy 
 Overall percentage of total prediction which model got right out of all test cases, In our case it was 88 percent for LRand SVC and 87 percent for RF, This is huge misleading for us cause Data set as approximately 80-90 0 valuses which suggest very rarely prediction was correct.

 ## Precision(Class - 1late)
 Higher preicion means lower false alarm, but In our case it was absolute 0 for all case so it's really bad suggesting model doesn't got right anything

 ## Recall/Sensivity(Class - 1late)
 Out of all true values i.e assignment that ended up being late how much did our model flag successfully, This show out all class which model predicted all were wrong

 ## F1 score
 Harmonic mean of Precision and Sensivity , turns out to be zero cause both precision and recall are zero suggesting models are entirely useless 

 ## AUC-ROC
 Area under the cureve of AUC-ROC decides whether model is better than random guessing or not, but since out values came out to be less <0.5 , All the above models were significanty worse than random guessing, also from knowledge pov It is used to train hyperparameter, out of all values RF was just above AUC-ROC aprrox 0.5064 which is still bad than random guessing

 # Model Selection and Architechure Analysis

 ## Best Model if forced to select??
 Honestly all models were entirely useless but when forced to select RandomForestClassifier comes to the top, it was marginally less broken(AUC_ROC = 0.51) than random baseline, while the other models were terrible and fell to 0.35

 ## Failure Analysis
Logistic Regression and SVC are mathematically driven to find a global boundary line that minimizes overall classification error. Because Class 0 heavily dominates, the loss optimization loop completely sacrificed the 40 late samples to maximize the accuracy score on the 284 on-time samples.
Random Forest splits data based on data purity. It struggled slightly less because it creates deep pathways, but without explicit class weights, it still defaulted to predicting the majority group.

# Practical Purpose 
If an operations team deployed this model today, it would serve no practical purpose and would be actively dangerous to rely on. Because its Class 1 Recall is 0.00, it would provide an institution with a false sense of security

# 100feature vs 15 features
Our model collapsed cause of severe class imbalance maximizing overall accuracy by simply guessing classes in prediction. Because of this models scored Catastropic 0 for class 1 prediction, recall which makes them completely blind for prediction and useless as a warning system. We need  better non linear signals for getting results as we expect, there was no change in 100 features models prediciton v/s top 15 features because all of these features didn't have sufficient information to learn from data and predict better than random baseline. 