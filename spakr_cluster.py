import datetime

from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import DecisionTreeClassifier
import pandas as pd
import re
import warnings
import os
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import datetime
import numpy as np
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel



import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.svm import SVC     ### SVM for classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
import math
from datetime import datetime
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.tree import RandomForest, RandomForestModel

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel

from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from datetime import datetime
from pyspark.ml.classification import FMClassifier


def run_with_spark():
        
  

    conf = SparkConf()
    conf.setMaster('spark://192.168.1.45:7077')
    conf.setAppName('spark-basic')
    conf.set("spark.executor.memory", "5g")
    conf.set("spark.executor.cores", "4")
    conf.set("spark.cores.max", "16") 
    conf.set("spark.driver.maxResultSize","16g") 
    
    sc = SparkContext(conf=conf)
    


    acc_DT =list()
    acc_NV = list()
    acc_RF = list()
    acc_GBT= list()



    time_DT =list()
    time_NV = list()
    time_RF = list()
    time_GBT = list()

    df= sc.textFile("/home/pc1/opt/data/data2M_10c.csv").map(lambda line: line.split(","))
    dataset  = df.map(lambda x: LabeledPoint(x[0], x[1:]))
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3])
    
    # decison treee
    start = datetime.now()
    model = DecisionTree.trainClassifier(trainingData,  numClasses=20, categoricalFeaturesInfo={}, impurity='gini', maxDepth=8, maxBins=32)

    
    
# Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))    
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    
    acc =0
    acc = (labelsAndPredictions.filter( lambda lp: lp[0] == lp[1]).count()) / float(testData.count())
    end = datetime.now() -start
    time_DT.append(end)
    acc_DT.append(acc)
    
    print("\n")
    print("Decison Tree")
    print ('time : ', end)
    print('Accuracy= ' + str(acc))


        # Train a naive Bayes model.
    start = datetime.now()
    model = NaiveBayes.train(trainingData, 1.0)
    
    predictionAndLabel = testData.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / testData.count()
    end = datetime.now() -start
    time_NV.append(end)
    acc_NV.append(accuracy)
    print("\n")
    print("Naive Bayes")
    print ('time : ', end)
    print('Accuracy = {} '.format(accuracy))

    



    


# ramdom forest
    numTrees = 10
    featureSubsetStrategy = "auto"
    maxDepth = 8
    maxBins = 32
    
    start = datetime.now()
    model = RandomForest.trainClassifier(trainingData, numClasses=20, categoricalFeaturesInfo={}, numTrees=numTrees, featureSubsetStrategy=featureSubsetStrategy, impurity='gini', maxDepth=maxDepth, maxBins=maxBins)
    
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    acc = labelsAndPredictions.filter(lambda lp: lp[0] == lp[1]).count() / float(testData.count())
    end = datetime.now() -start
    time_RF.append(end)
    acc_RF.append(acc)
    print("\n")
    print("Random Forest")
    print ('time : ', end)
    print('Accuracy= ' + str(acc))
    
    
#  gradient boosted tree
    start = datetime.now()
    
    model = GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo={}, numIterations=10)
    
    
    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    
    acc = labelsAndPredictions.filter(lambda lp: lp[0] == lp[1]).count() / float(testData.count())
    end = datetime.now() - start
    time_GBT.append(end)
    acc_GBT.append(acc)
    print("\n")
    print("Gradient Boosted Trees")
    print ('time : ', end)
    print('Accuracy = {} '.format(acc))



    # results =[]
    # results.append(acc_DT)
    # results.append(acc_NV)
    # results.append(acc_RF)
    # results.append(acc_GBT)

    # names =('Decision tree','Navie bayes','Random forest','Gradient Boosted Tree')
    # fig = plt.figure()
    # fig.suptitle('Algorithm Comparison')
    

    # plt.boxplot(results, labels=names)
    # plt.ylabel('Accuracy')    

    # #ax.set_xticklabels(names)
    # plt.show()   

    print("Kết quả \n")
    print("Decision Tree")
    print("Acc:",acc_DT[0])
    print("Time:",time_DT[0])

    print("\n")
    print("Naive Bayes ")
    print("Acc:",acc_NV[0])
    print("Time:",time_NV[0])

    print("\n")
    print("Random Forest ")
    print("Acc:",acc_RF[0])
    print("Time:",time_RF[0])


    print("\n")
    print("Gradient Boosted Tree")
    print("Acc:",acc_GBT[0])
    print("Time:",time_GBT[0])

    





def main():
    
    run_with_spark()
if __name__== '__main__':
    main()
    
