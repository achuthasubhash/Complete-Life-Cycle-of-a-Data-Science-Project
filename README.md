# Complete-Life-Cycle-of-a-Data-Science-Project

1.Data collection

a.web scraping  best article to refer-https://towardsdatascience.com/choose-the-best-python-web-scraping-library-for-your-application-91a68bc81c4f

   1.beautifulsoup
   
   2.scrapy
   
   3.selenium
   
   4.request to access data 
  
b.3rd party API'S 

c.big data engineering to collect data

d.databases

e.free online resource

   1)kaggle
   
   2)movielens
   
   3)data.gov:https://data.gov.in/
   
   4)uci
   
   5)quandi
   
   6)world3bank  https://data.world/
   
   7)UCIMachineLearning
   
   8)online hacktons
   
   9)image data from Google_Search
   
   10)image data from Bing_Search
   
   11)https://www.columnfivemedia.com/100-best-free-data-sources-infographic
   
   12)Reddit:https://lnkd.in/dv5UCD4
   
   13)https://datasets.bifrost.ai/?ref=producthunt
   
   14)data.world:https://lnkd.in/gEK897K
   
   15)https://data.world/datasets/open-data
   
   16)FiveThirtyEight :-  https://lnkd.in/gyh-HDj
   
   17)BuzzFeed :- https://lnkd.in/gzPWyHj
   
   18)Google public datasets :- https://lnkd.in/g5dH8qE
   
   19)Quandl :- https://www.quandl.com
   
   20)socorateopendata :- https://lnkd.in/gea7JMz
   
   21)AcedemicTorrents :- https://lnkd.in/g-Ur9Xy
   
   22)labelimage
   
   23)tensorflow_datasets as tfds
   
   24)https://datasets.bifrost.ai/?ref=producthunt
   
   25)https://ourworldindata.org/
   
   26)https://data.worldbank.org/
   
   27)google open images:https://storage.googleapis.com/openimages/web/download.html

2.Feature engineering

   a.handle missing value
   
     1.if missing data too small then  delete row
     
     2.replace mean,median,mode
     
     3.apply classifier algorithm to predict missing value
     
     4.knn imputer
     
     5.apply unsupervised 
     
     6.Random Sample Imputation
     
     7.Missing Data Not At Random
     
     8.Missing At Random
     
     9.Adding a variable to capture NAN
     
   b.handle imbalance
   
     1.Under Sampling
     
     2.Over Sampling  (RandomOverSampler (here new points create by same dot)) ,  SMOTETomek(new points create by nearest point so take long time)
     
     3.class_weight give more importance to that small class
  
   c.remove noise
   
   d.format data
   
   e.handle categorical data
   
     1.One Hot Encoding
     
     2.Count Or Frequency Encoding
     
     3.Target Guided Ordinal Encoding
     
     4.Mean Encoding
     
     5.Probability Ratio Encoding
     
   f.normalisation of data
   
     1.Standardization
     
     2.Min Max Scaling
     
     3.Robust Scaler
     
     4.Q-Q plot check whether feature is guassian or normal distributed
     
        a.Guassian Transformation
        
        b.Logarithmic Transformation
        
        c.Reciprocal Trnasformation
        
        d.Square Root Transformation
        
        e.Exponential Transdormation
        
        f.BoxCOx Transformation
        
        g.log(1+x)
        
   g.remove low variance data 
   
   h.same variable in feature then remove

3.eda

4.Feature selection

  1.pearson correleation
  
  2.heatmap
  
  3.extra tree classifier (Feature Importance)
  
  4.staatics to select important feature
  
  5.keep in mind of curse of dimensionality
  
  6.highly correleated then remove 1 feature
  
  7.Feature importance
  
  8.dimension reduction
  


5.Model

select right model

hyperparameter 


6.Test

test 

if not good performance go back to Data collection or  Feature engineering


7.deployment

azure,flask,aws,gcp


8.mointoring model
