# Complete-Life-Cycle-of-a-Data-Science-Project

***CREDITS:All corresponding resources***

***MOTIVATION:Motivation to create this repository  to help upcoming aspirants and help to  others in the data science field***

***Business understanding*** 

***1.Data collection***

Data consists of 3 kinds

    a.Structure data (tabular data,etc...)
    
    b.Unstructured data (images,text,audio,etc...)
    
    c.semi structured data (XML,JSON,etc...)
    
variable 
   
    a.qualitative (nominal,ordinal,binary)
    
    b.quantitative(discrete,continuous)

a.Web scraping  best article to refer-https://towardsdatascience.com/choose-the-best-python-web-scraping-library-for-your-application-91a68bc81c4f

https://www.analyticsvidhya.com/blog/2019/10/web-scraping-hands-on-introduction-python/?utm_source=linkedin&utm_medium=KJ|link|weekend-blogs|blogs|44087|0.875

    1.Beautifulsoup
   
    2.Scrapy
   
    3.Selenium
   
    4.Request to access data 
    
    5.AUTOSCRAPER - https://github.com/alirezamika/autoscraper
    
    6.Twitter scraping tool (𝚝𝚠𝚒𝚗𝚝 or tweepy)-https://github.com/twintproject/twint
    
      https://analyticsindiamag.com/complete-tutorial-on-twint-twitter-scraping-without-twitters-api/
    
    7.urllib
    
b.Web Crawling
  
b.3rd party API'S 

c.creating own data  (manual collection eg:google docx,servey,etc...) primary data

d.Databases

  Databases are 2 kind sequel  and no sequel database

  sql,sql lite,mysql,mongodb,hadoop,elastic search,cassendra,amazon s3,hive,googlebigtable,AWS DynamoDB,HBase,oracle db

e.Online resources -   ultimate resource  https://datasetsearch.research.google.com/

    1)kaggle-https://www.kaggle.com/datasets
   
    2)movielens-https://grouplens.org/datasets/movielens/latest/
   
    3)data.gov-https://data.gov.in/
   
    4)uci-https://archive.ics.uci.edu/ml/datasets.php
   
    5)Group Lens dataset
    
    6)world3bank  https://data.world/ , worldbank
   
    7)Google Cloud BigQuery public datasets
   
    8)online hacktons
   
    9)image data from google_images_download
   
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
   
    22)labelimage:- https://github.com/wkentaro/labelme  ,  https://github.com/tzutalin/labelImg
   
    23)tensorflow_datasets as tfds
   
    24)https://datasets.bifrost.ai/?ref=producthunt
   
    25)https://ourworldindata.org/
   
    26)https://data.worldbank.org/
   
    27)google open images:https://storage.googleapis.com/openimages/web/download.html
   
    28)https://data.gov.in/
    
    29)imagenet dataset-http://www.image-net.org/
    
    30)https://parulpandey.com/2020/08/09/getting-datasets-for-data-analysis-tasks%e2%80%8a-%e2%80%8aadvanced-google-search/
    
    31)https://storage.googleapis.com/openimages/web/index.html  , 
    
       https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=segmentation&r=false&c=%2Fm%2F09qck
     
    32)coco dataset https://cocodataset.org/#explore
    
    33)huggingface datasets-https://github.com/huggingface/datasets
    
    34)Big Bad NLP Database-https://datasets.quantumstat.com/
    
    35)https://www.edureka.co/blog/25-best-free-datasets-machine-learning/
    
    36)bigquery public dataset ,Google Public Data Explorer
    
    37)inbuilt library data eg:iris dataset,mnist dataset,etc...
    
    38)data.gov.be ,data.egov.bg/ ,data.gov.cz/english ,portal.opendata.dk,govdata.de,opendata.riik.ee,data.gov.ie,data.gov.gr,datos.gob.es,data.gouv.fr,data.gov.hr
    
    dati.gov.it,data.gov.cy,opendata.gov.lt,data.gov.lv,data.public.lu,data.gov.mt,data.overheid.nl,data.gv.at,danepubliczne.gov.pl,dados.gov.pt,data.gov.ro,podatki.gov.si

    data.gov.sk,avoindata.fi,oppnadata.se,https://data.adb.org/ ,https://data.iadb.org/ ,https://www.weforum.org/agenda/2018/03/latin-america-smart-cities-big-data/
    
    https://data.fivethirtyeight.com/ , https://wiki.dbpedia.org/ ,https://www.europeandataportal.eu/en ,https://data.europa.eu/ ,https://www.census.gov/,
    
    https://www.who.int/data/gho ,https://data.unicef.org/open-data/ ,http://data.un.org/ ,https://data.oecd.org/ ,https://data.worldbank.org/  
    
    39.Awesome Public Dataset-https://github.com/awesomedata/awesome-public-datasets
    
    40.Datasets for Machine Learning on Graphs-https://ogb.stanford.edu/
    
    41.Big Bad NLP Database-https://datasets.quantumstat.com/
    
    42.30 largest tensorflow datasets-https://lionbridge.ai/datasets/tensorflow-datasets-machine-learning/
    
    43.50+ Object Detection Datasets-https://medium.com/towards-artificial-intelligence/50-object-detection-datasets-from-different-industry-domains-1a53342ae13d
       
***2.Feature engineering***

   Data cleaning-Pyjanitor-https://analyticsindiamag.com/beginners-guide-to-pyjanitor-a-python-tool-for-data-cleaning/
     
   Remove duplicate data in dataset

   a.Handle missing value
   
     Types of missing value 
     
     1.missing completely at random(no correlation b/w missing and observed data) we can delete no disturbance of data distribution
     
     2.missing at random (randomness in missing data, missing value have correlation by data) we can't delete because disturbance of data distribution
     
     3.missing not at random  (there is reason for missing value and directly related to value)
   
     1.if missing data too small then delete it 
     
     2.replace by statistical method mean(influenced by outiler),median(not influenced by outiler),mode
     
     3.apply classifier algorithm to predict missing value
     
     4.knn imputer
     
     5.apply unsupervised 
     
     6.Random Sample Imputation
     
     7.Adding a variable to capture NAN
     
     8.Arbitrary Value Imputation
     
     9.hot deck Imputation
     
     10.regression Imputation
     
     11.End of Distribution Imputation
     
     12.Missing indicator
     
   b.Handle imbalance
   
     1.Under Sampling - mostly not prefer because lost of data
     
     2.Over Sampling  (RandomOverSampler (here new points create by same dot)) ,  SMOTETomek(new points create by nearest point so take long time)
     
     3.class_weight give more importance(weight) to that small class
     
     4.use Stratified kfold to keep the ratio of classess constantly
  
   c.Remove noise data
   
   d.Format data
   
   e.Handle categorical data   Ordinal,Nominal,cyclic,binary categorical variables  
   
     1.One Hot Encoding
     
     2.Count Or Frequency Encoding
     
     3.Target Guided Ordinal Encoding
     
     4.Mean Encoding
     
     5.Probability Ratio Encoding
     
     6.label encoding
     
     7.probability ratio encoding
     
     8.woe(Weight_of_evidence)
     
     9.one hot encoding with multi category (keep most frequently repeated only)
     
     10.feature hashing 
     
     11.sparse csr matrix
     
     12.entity embeddings
     
     13.binary encoding
     
     14.Rare label encoding
     
   f.Scaling of data
   
       1.Normalisation (Min Max Scaling) robust scaling
   
       2.Standardization
     
       3.Robust Scaler not influenced by outliers because using of median,IQR
       
       4.Mean normalization
       
       5.maximum absolute scaling
      
   Q-Q plot or Shapiro-Wilk Normality Test  is used to check whether feature is guassian or normal distributed  required for linear regression,logistic regression to Improve 
performance if not distributed then use below methods to bring it guassian distribution
     
           a.Guassian Transformation
        
           b.Logarithmic Transformation
        
           c.Reciprocal Trnasformation
        
           d.Square Root Transformation
        
           e.Exponential Transdormation
        
           f.BoxCOx Transformation
        
           g.log(1+x) Transformation
           
           h.johnson
        
   g.Remove low variance feature by using VarianceThreshold
   
   h.Same variable(only 1 variable) in feature then remove feature
   
   i.Outilers   removing outilers depond on problem we are solving
   
      2 type of outilers available: Global outiler, Local outiler
    
      eg: incase of fraud detection outilers are very important
      
      methods to find outiler: zscore,boxplot,scatter plot,IQR,TensorFlow_Data_Validation
      
      Automatic Outlier Detection:Isolation Forest,Local Outlier Factor,Minimum Covariance Determinant
      
      if outiler present then use robust scaling
      
   j.Anomaly
   
     clustering techniques to find it
      
   k.Sampling techniques
     
     a.biased sampling
     
     b.unbiased sampling

***3.Exploratory Data Analysis(eda)***
  
    Explore the dataset by using  python or microsoft excel or tableau or powerbi, etc...
    
    Data visualization (Matplotlib,Seaborn,Bokeh,ggplot,visualizer,etc...)
    
    Scatterplot,multi line plot,bubble chart,bar chart,histogram,boxplot,distplot,index plot,violin plot,time series plot,density plot,dot plot,strip plot,plotly,Choropleth Map,PDF,Kernel density function,networkx
  
    univariate and bivariate and multivariate analysis
    
    model visualization Tensorboard,netron,playground tensorflow,plotly
    
    distributions(discerte,continous)
    
    data distributions-normal distribution,Standard Normal Distribution,Student's t-Distribution,Bernoulli Distribution,Binomial Distribution,Poisson Distribution,Uniform Distribution,F Distribution,Covariance and Correlation
  
    Types of Statistics  
    
    1.Descriptive
    
    2.Inferential
    
    Types of data
    
    1) Categorical (nomial,ordinal)
     
    2) Numerical   (discerte,continous)
    
    random variable(discerte random variable ,continous random variable)
    
    Central Limit Theorem,Bayes Theorem,Confidence Interval,Hypothesis Testing,z test, t test,f test,Confidence Interval,1 tail test, 2 tail test,chisquare test,anova test,A/B testing
  
***4.Feature selection***

    1.pearson correleation
   
    2.chisquare,Anova
  
    3.Feature Importance
  
       a.ExtraTreesClassifier,ExtraTreesregressor
    
       b.SelectKBest
    
       c.Logistic Regression
    
       d.Random_forest_importance
       
       e.decision tree
       
       f.Linear Regression
       
    4.statics to select important feature (chi square  test,T test,anova test,hypothesis test,ANOVA)
  
    5.keep in mind  curse of dimensionality (as dimension increases performance decreases)
  
    6.highly correleated then can remove 1 feature (multicollinearity)
  
    7.dimension reduction
  
    8.lasso regression to penalise unimportant features
    
    9.filter method,wrapper method,embedded method
    
    10.threshold based method 
    
    11.hypothesis testing
    
    12.model based selection
    
    13.Mutual Information Feature Selection
    
    14.Correlation Feature Selection
    
    15.remove features with very low variance (quasi constant feature dropping)
    
    16.Univariate  feature selection
    
    17.recursive feature  elimination,recursive feature addition,Exhaustive search
    
    18.importance of feature (random forest importance)
    
    19.feature importance with decision trees
    
    20.Step forward feature selection,Step backward feature selection
    
    21.PyImpetus
    
    22.drop constant features (variance=0)
    
***5.Data splitting***

     Splitting ratio of data deponds on size of dataset available

     Training data,Validation data,Testing data

***6.Model selection***

  Machine learning
   
   A.Supervised learning (have label data)
   
     1.Regression (output feature in continous data form)
     
       linear regression,polynomial regression,support vector regression,Decision Tree Regression,Random Forest Regression,
       
       least square method,Random Forest Regression,xgboost,ridge(L2 Regularization),lasso(L1 Regularization),catboost,gradientboosting,adaboost,
       
       elsatic net,light gbm,ordinary least squares,cart
       
       use cases:
   
     2.Classification (output feature in categorical data form)
     
        Logistic Regression,K-Nearest Neighbors,Support Vector Machine,Kernel SVM,Naive Bayes,Decision Tree Classification,
        
        Random Forest Classification,xgboost,adaboost,catboost,gaussian NB,LGBMClassifier,LinearDiscriminantAnalysis,
        
        passive aggressive classifier algorithm,cart,c4.5,c5.0
        
        use cases:
   
   B.Unsupervised learning(no label(target) data)
   
     1.Dimensionality reduction - PCA,SVD,LDA,tsne,plsr,pcr,autoencoders,kpca
   
     2.Clustering :https://scikit-learn.org/stable/modules/clustering.html
   
     3.Association Rule Learning - support,lift,confidence,aprior,elcat,Fp-growth,Fp-tree construction
   
     4.Recommendation system -
     
         a.collaborative Recommendation system (model based, memory based)
         
         b.content based Recommendation system 
         
         c.utility based Recommendation system 
         
         d.knowledge based Recommendation system 
         
         e.demographic based Recommendation system 
         
         f.hybrid based Recommendation system 
         
         g.Average Weighted Recommendation
         
         h.using K Nearest Neighbor
         
         i.cosine distance recommender system
         
         j.TensorFlow Recommenders
         
         k.suprise baseline model
   
   C.Ensemble methods
   
     1.Stacking models
   
     2.Bagging models
   
     3.Boosting models
     
     4.Blending
     
     5.Voting (Hard Voting,Soft Voting)
   
   D.Reinforcement learning
   
      2 types a)model free   b)model based
   
      agent,environment,policy,reward function,value function,state,action,episode
   
      agent apply action to environment get corresponding reward so that it learn environment
      
      1.Q-Learning
      
      2.Deep Q-Learning
      
      3.Deep Convolutional Q-Learning
      
      4.Twin Delayed DDPG
      
      5.A3C 
      
      6.Advantage weighted actor critic (AWAC). 
      
      7.XCS
      
      https://simoninithomas.github.io/deep-rl-course/
   
   E.Deep-learning  (use when have huge data and data is highly complex and state of art for unstructured data)
   
   Frameworks:Pytorch,Tensorflow,Keras,caffe,theano
   
   1.Multilayer perceptron(MLP)
   
     1.Regression task
   
     2.Classification task
   
   2.Convolutional neural network ( use for image data)
   
     1.Classification of image
     
       create own model,Lenet,Alexnet,Resenet,Inception,Vgg,Efficient,Nasnet
     
     2.Localization of object in image
     
     3.Object detection and object segmentation 
     
       rcnn,fastrcnn,fatercnn,yolo v1,yolo v2,yolo v3,yolo v4,fast yolo,yolo tiny,yolo lite,yolo tiny++,yolo act++,
       
       maskrcnn,ssd,detectron,detectron2,mobilenet,retinanet,R-fcn,detr facebook,U-net
       
       3 kind of object segmentation are available semantic segmentation,instance segmentation,panoptic segmentation
     
     4.DeepSORT,Pose estimation 
     
     5.Deepdream,Neural style transfer
     
     CNNs 'see' - FilterVisualizations, Heatmaps,Saliency Maps,Heat Map Visualizations
     
     Data Augmentation apply to increase size of dataset and performance of model
   
   3.Recurrent neural network (use when series of data)
   
     1.RNN
     
     2.GRU
     
     3.LSTM (have memory cell,forget gate  etc..)
     
     all above 3 models have bidirectional also based on problem statement use bidirectional model
  
   4.Generative adversarial network 
   
     Cycle gan,Dcgan,SRGAN,InfoGAN,stargan,attan gan,stylegan,,PixelRNN,DiscoGAN,lsGAN
   
   5.Autoencoder
   
      1.sparse Autoencoder
      
      2.denoising Autoencoder
      
      3.Contractive Autoencoder
      
      4.stacked Autoencoder
      
      5.deep Autoencoder
      
      6.variational autoencoder
   
   6.BoltzmannMachines,deep belief network,deep BoltzmannMachines
   
   7.Self Organizing Maps (SOM)
   
   8.Natural language processing
   
     Clean data(removing stopwords depond on problem ,lowering data,tokenization,postagging,stemmimg or lemmatization depond on problem,skipgram,n-gram,chunking)
     
     Nltk,spacy,genism,textblob,inltk,stanza,polygot,corenlp,polyglot,PyDictionary,Huggiing face,spark nlp,allen nlp,rasa nlu,Megatron,texthero  libraries
     
     NLU,NLG,NER,text summarization,Sentiment Analysis,Text Classifications,machine translation,chat bot
      
     1.bag of words
     
     2.Tfidf
     
     3.using rnn,lstm,gru
     
       for above 3 models have bidirectional also
     
     4.Encoder and Decoder(sequence to sequence)
     
     5.wordembedding
        
        a.using pretrained model 
          
          i)word2vec( cbow,skipgram)
          
          ii)glove
          
          iiI)fasttext
        
        b.creating own embedding  (use when have huge data)
        
          i)word2vec library
          
          ii)keras embedding 
        
     6.Document embedding-Doc2vec
      
     7.sentence embedding
    
       sense2vec,SENT2VEC,Universal sentence encoder
      
     8.attention
     
     9.self attention
     
     10.Transformer (big breakthrough in NLP) - http://jalammar.github.io/illustrated-transformer/  
      
     11.BERT,Quantized MobileBERT,ALBERT,ELMo,ROBERTA,XLNet,XLM-RoBERTa,T5,DISTILBERT,GPT,GPT2,GPT3,PRADO,PET
     
        http://jalammar.github.io/    http://jalammar.github.io/illustrated-bert/   http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
   
   F.Time Series 
   
      here data split is different (train,test,validate)
      
      here handling missing data different 
      
      generally used  to impute data in Time Series
      
      1.ffill
      
      2.bfill
      
      3.do mean of previous or future x samples and impute
      
      4.take previous year value and impute
      
      here model selection deponds on different property of data like stationary,trend,seasonality,cyclic
      
      adfuller test  for  Stationarity
      
      models 
      
      1.Arima , auto arima ,seasonal arima
      
      2.Autoregressive 
      
      3.Moving average,Exponential Moving average,Exponential Smoothing
      
      4.Lstm(neural network)
      
      5.Autoregressive
      
      6.Navie forecasts
      
      7.Smoothing (moving average,exponential smoothing)
      
      8.Facebook prophet (note:expceted date column as ds and target column as y)
      
      9.Holts winter,Holts linear trend
      
      10.AutoTS-https://analyticsindiamag.com/hands-on-guide-to-autots-effective-model-selection-for-multiple-time-series/
      
      11.Temporal Convolutional Neural
      
      12.Atspy For Automating The Time-Series Forecasting-https://analyticsindiamag.com/hands-on-guide-to-atspy-for-automating-the-time-series-forecasting/
      
      13.Darts-https://analyticsindiamag.com/hands-on-guide-to-darts-a-python-tool-for-time-series-forecasting/
      
      14.Bayesian Neural Network 
      
      15.PyFlux-https://analyticsindiamag.com/pyflux-guide-python-library-for-time-series-analysis-and-prediction/
      
      best article-https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/,
      
      https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
      
      https://github.com/Apress/hands-on-time-series-analylsis-python
          
   G.Semi supervised learning,Self-Supervised Learning,Multi-Instance Learning
   
   H.Active learning,Multi-Task Learning,Online Learning
   
   I.Transfer learning(Inductive Transfer learning(similar domain,different task),Unsupervised Transfer Learning(different task,different domain but similar enough) ,Transductive Transfer Learning(similar task,different domain))
   
   https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a
   
   J.Deep dream,Style transfer
   
   K.One-shot learning,Zero-shot learning

***Hyperparameter tuning***
  
    a.GridSearchCV (check every given parameter so take long time)
  
    b.RandomizedSearchCV (search randomly narrow down our time)
  
    c.Bayesian Optimization (Hyperopt)
    
    d.Sequential Model Based Optimization(Tuning a scikit-learn estimator with skopt)

    e.Optuna
  
    f.Genetic Algorithms
    
    g.Keras tuner
    
    https://towardsdatascience.com/10-hyperparameter-optimization-frameworks-8bc87bc8b7e3
    
 Cross validation techniques- https://towardsdatascience.com/understanding-8-types-of-cross-validation-80c935a4976d
    
     1.Loocv
     
     2.Kfoldcv
     
     3.Stratfied cross validation
     
     4.Time Series cross-validation
     
     5.Holdout cross-validation
     
     6.Repeated cross-validation
  
Tensorboard to visualization of model performance

Distributed Training with TensorFlow 

***6.Testing model***

Generally used metrics
  
     Always check bias variance tradeoff to know how model is performing
     
     Model can be overfitting(low bias,high variance),underfitting(high bias,low variance),good fit(low bias,low variance)
     
    1.Regression task - mean-squared-error, Root-Mean-Squared-Error,mean-absolute error, R², Adjusted R²,Cross-entropy loss,Mean percentage error 
   
    2.Classification task-Accuracy,confusion matrix,Precision,Recall,F1 Score,Binary Crossentropy,Categorical Crossentropy,AUC-ROC curve,log loss,Average precision,Mean average precision
    
    3.Reinforcement learning - generally  use rewards
    
    4.Incase of machine translation use bleu score
    
    5.Clustering then use silhouette score
    
    6.Object Detection loss-localization loss,classification loss,Focal Loss,IOU,L2 loss


Docker and Kubernetes

***7.deployment***

    1.Azure
    
    2.Heroku
    
    3.Amazon Web Services
    
    4.Google cloud platform
    
    MODEL DEPLOYMENT USING TF SERVING
    
    Models visualization using Tensorboard,netron

    Python Frameworks for App Development- Flask,Streamlit,Django,Web2py,Pyramid,CherryPy,Voila https://analyticsindiamag.com/top-8-python-tools-for-app-development/
    
    Web-Based GUI (Gradio)- https://analyticsindiamag.com/guide-to-gradio-create-web-based-gui-applications-for-machine-learning/
    
***Tensorflow lite:Use of tensorflow lite to reduce size of model***
    
***Quantization:Use Quantization to reduce size of model***

***8.Mointoring model***

***CI CD pipeline used-  circleci , jenkins***

***In real world project use pipeline*** -https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
   
   1.easy debugging
   
   2.better readability
   
***BIG DATA: hadoop,apache spark***

***research paper***-https://arxiv.org/ ,  https://www.kaggle.com/Cornell-University/arxiv

   code for Research Papers-https://chrome.google.com/webstore/detail/find-code-for-research-pa/aikkeehnlfpamidigaffhfmgbkdeheil

***programming language for data science is Python, R,Julia,Java,Scala*** 

   IDE:jupyter notebook,spyder,pycharm,visual studio

***BEST ONLINE COURSES***

    1.COURSERA

    2.UDEMY

    3.EDX

    4.DATACAMP
    
    5.Udacity

***BEST YOUTUBE CHANNEL TO FOLLOW***

    1.Krish Naik-https://www.youtube.com/user/krishnaik06
    
    2.Codebasics-https://www.youtube.com/channel/UCh9nVJoWXmFb7sLApWGcLPQ  

    3.Abhishek thakur-https://www.youtube.com/user/abhisheksvnit

    4.AIEngineering-https://www.youtube.com/channel/UCwBs8TLOogwyGd0GxHCp-Dw

    5.Ineuron-https://www.youtube.com/channel/UCb1GdqUqArXMQ3RS86lqqOw
    
    6.Ken jee-https://www.youtube.com/c/KenJee1/featured       
    
    7.3Blue1Brown-https://www.youtube.com/c/3blue1brown/featured
    
    8.The AI Guy -https://www.youtube.com/channel/UCrydcKaojc44XnuXrfhlV8Q 

***BEST BLOGS TO FOLLOW***

    1.Towards data science-https://towardsdatascience.com/

    2.Analyticsvidhya-https://www.analyticsvidhya.com/blog/?utm_source=feed&utm_medium=navbar

    3.Medium-https://medium.com/
    
    4.Machinelearningmastery-https://machinelearningmastery.com/blog/

***BEST RESOURCES***

   1.paperswithcode-https://paperswithcode.com/methods

   2.madewithml-https://madewithml.com/topics/   Weights & Biases-https://wandb.ai/gallery

   3.Deep learning-https://course.fullstackdeeplearning.com/#course-content

   4.pytorch deep learning-https://atcold.github.io/pytorch-Deep-Learning/

   5.deep-learning-drizzle-https://deep-learning-drizzle.github.io/  https://deep-learning-drizzle.github.io/index.html

   6.Fastaibook-https://github.com/fastai/fastbook
    
   7.TopDeepLearning-https://github.com/aymericdamien/TopDeepLearning
   
   8.NLP-progress-https://github.com/sebastianruder/NLP-progress
    
   9.EasyOCR-https://github.com/JaidedAI/EasyOCR
    
   10.Awesome-pytorch-list-https://github.com/bharathgs/Awesome-pytorch-list
    
   11.free-data-science-books-https://github.com/chaconnewu/free-data-science-books
    
   12.arcgis-https://github.com/Esri/arcgis-python-api 
    
   13.data-science-ipython-notebooks-https://github.com/donnemartin/data-science-ipython-notebooks
    
   14.julia-https://github.com/JuliaLang/julia  , https://docs.julialang.org/en/v1/
    
   15.google-research-https://github.com/google-research/google-research
    
   16.reinforcement-learning-https://github.com/dennybritz/reinforcement-learning
    
   17.keras-applications-https://github.com/keras-team/keras-applications  ,  https://github.com/keras-team/keras
    
   18.opencv-https://github.com/opencv/opencv
    
   19.transformers-https://github.com/huggingface/transformers
    
   20.code implementations for research papers-https://chrome.google.com/webstore/detail/find-code-for-research-pa/aikkeehnlfpamidigaffhfmgbkdeheil
    
   21.regarding satellite images
   
       ersi arcgis-https://www.esri.com/en-us/arcgis/about-arcgis/overview
       
       earthcube-https://www.earthcube.eu/
   
   22.Monk_Object_Detection-https://github.com/Tessellate-Imaging/Monk_Object_Detection
   
   23.NLP-progress - https://github.com/sebastianruder/NLP-progress
   
   24.interview-question-data-science-https://github.com/iNeuronai/interview-question-data-science-
   
   25.recommenders-https://github.com/microsoft/recommenders
   
   26.Awesome-NLP-Resources -https://github.com/Robofied/Awesome-NLP-Resources
   
   27.Tool for visualizing attention in the Transformer model-https://github.com/jessevig/bertviz
   
   28.TransCoder-https://github.com/facebookresearch/TransCoder
   
   29.Tessellate-Imaging-https://github.com/Tessellate-Imaging/monk_v1
   
   Monk_Object_Detection-https://github.com/Tessellate-Imaging/Monk_Object_Detection/tree/master/application_model_zoo
   
   Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials- https://github.com/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials
   
   30.Machine-Learning-with-Python-https://github.com/tirthajyoti/Machine-Learning-with-Python
   
   31.huggingface contain almost all nlp pretrained model and all tasks related to nlp field
     
      https://github.com/huggingface  https://github.com/huggingface/transformers    https://huggingface.co/transformers/
   
   32.multi-task-NLP-https://github.com/hellohaptik/multi-task-NLP
   
   33.gpt-2 - https://github.com/openai/gpt-2
   
   34.Powerful and efficient Computer Vision Annotation Tool (CVAT)-https://github.com/openvinotoolkit/cvat, https://github.com/abreheret/PixelAnnotationTool
   
   https://github.com/UniversalDataTool/universal-data-tool
   
   35.Data augmentation for NLP-https://github.com/makcedward/nlpaug
   
   36.awesome Data Science-https://github.com/academic/awesome-datascience
   
   37.mlops-https://github.com/visenger/awesome-mlops
   
   38.gym-https://github.com/openai/gym
   
   39.Super Duper NLP Repo-https://notebooks.quantumstat.com/
   
   40.papers summarizing the advances in the field-https://github.com/eugeneyan/ml-surveys
   
   41.deep-translator-https://github.com/nidhaloff/deep-translator
   
   42.detext-https://github.com/linkedin/detext
   
   43.nlpaug-https://github.com/makcedward/nlpaug
   
   44.ipython-sql-https://github.com/catherinedevlin/ipython-sql
   
   45.libra-https://github.com/Palashio/libra
   
   46.opencv-https://github.com/opencv/opencv
   
   47.learnopencv-https://github.com/spmallick/learnopencv  ,  https://www.learnopencv.com/
   
   48.math is fun-https://www.mathsisfun.com/  , https://pabloinsente.github.io/intro-linear-algebra, https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/
   
   49.DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ - https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
    
   50.Spark Release 3.0.1-https://spark.apache.org/releases/spark-release-3-0-1.html
   
   51.for more cheatsheets-https://github.com/FavioVazquez/ds-cheatsheets  , https://medium.com/swlh/the-ultimate-cheat-sheet-for-data-scientists-d1e247b6a60c
   
   52.text2emotion-https://pypi.org/project/text2emotion/
   
   53.ExploriPy-https://analyticsindiamag.com/hands-on-tutorial-on-exploripy-effortless-target-based-eda-tool/
   
   54.TCN-https://github.com/philipperemy/keras-tcn
   
   55.deeplearning-models-https://github.com/rasbt/deeplearning-models
   
   56.earthengine-py-notebooks-https://github.com/giswqs/earthengine-py-notebooks
   
   57.NLP-progress -https://github.com/sebastianruder/NLP-progress
   
   58.numerical-linear-algebra -https://github.com/fastai/numerical-linear-algebra
   
   59.Super Duper NLP Repo- https://notebooks.quantumstat.com/
   
   60.reinforcement learning by using  PyTorch-https://github.com/SforAiDl/genrl
   
   61.chatbot- from scratch,google dialogflow,rasa nlu,azure luis,Amazon lex,Wit.ai,Luis.ai,IBM Watson  etc...
   
   62.Teachable Machine-https://teachablemachine.withgoogle.com/
   
   64.tensorflow development-https://blog.tensorflow.org/
   
   63.Data Science in the Cloud-Amazon SageMaker,Amazon Lex,Amazon Rekognition,Azure Machine Learning (Azure ML) Services,Azure Service Bot framework,Google Cloud AutoML
   
   64.platforms to build and deploy ML models -Uber has Michelangelo,Google has TFX,Databricks has MLFlow,Amazon Web Services (AWS) has Sagemaker
   
   65.Time Complexity Of Machine Learning Models -https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/
  
   66.ML from scratch-https://dafriedman97.github.io/mlbook/content/introduction.html
   
   67.turn-on visual training for most popular ML algorithms https://github.com/lucko515/ml_tutor  https://pypi.org/project/ml-tutor/
   
   68.mlcourse.ai is a free online- https://mlcourse.ai/
   
   69.using pretrained model provided by tfhub- https://tfhub.dev/
   
   70.Deep-Learning-with-PyTorch- https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf
   
   71.MIT 6.S191 Introduction to Deep Learning-http://introtodeeplearning.com/
   
   72.R for Data Science-https://r4ds.had.co.nz/ ,Fundamentals of Data Visualization-https://clauswilke.com/dataviz/
   
   73.MAGENTA-https://magenta.tensorflow.org/
   
***Follow leaders in the field to update yourself in the field***

    1.Linkedin

    2.Twitter
    
***Free CPU/GPU/TPU***
 
    1.Google cloab
    
    2.Kaggle kernel(read terms and conditions before use)
    
    3.Paperspace Gradient(read terms and conditions before use)
    
    4.knime - https://www.knime.com/(read terms and conditions before use)
    
    5.RapidMiner (read terms and conditions before use)
    
***So what next ?***

participate online competition and do project and apply to intership ,job,solving real world problems, etc...

applications of data science in many industry

    1.E-commerce- Identifying consumers,Recommending Products,Analyzing Reviews
    
    2.Manufacturing- Predicting potential problems,Monitoring systems,Automating manufacturing units, Maintenance Scheduling,Anomaly Detection
    
    3.Banking- Fraud detection,Credit risk modeling,Customer lifetime value
    
    4.Healthcare- Medical image analysis, Drug discovery,Bioinformatics,Virtual Assistants,image segmentation
    
    5.Transport- Self-driving cars,Enhanced driving experience,Car monitoring system,Enhancing the safety of passengers
    
    6.Finance- Customer segmentation,Strategic decision making,Algorithmic trading,Risk analytics
    
    7.Marketing (Added from comments Credits: Jawad Ali)- LTV predictions,Predictive analytics for customer behavior,Ad targeting
    
    and many more fields
    
    
***online competitions:***

1.Kaggle-https://www.kaggle.com/

2.hackerearth-https://www.hackerearth.com/challenges/

3.machinehack-https://www.machinehack.com/

4.analyticsvidhya-https://datahack.analyticsvidhya.com/contest/all/

5.zindi-https://zindi.africa/competitions

6.crowdai-https://www.crowdai.org/

7.driven data-https://www.drivendata.org/

8.dockship-https://dockship.io/

9.International Data Analysis Olympiad (IDAHO)

10.Codalab

11.Iron Viz

12.Data Science Challenges

13.Tianchi Big Data Competition

***Some useful content :***

1. H20.ai automl, google automl,Azure Cognitive Services,Google Cloud Platform

2. Tpot

3. autopandas

4. AutoGluon   https://analyticsindiamag.com/how-to-automate-machine-learning-tasks-using-autogluon/

5. autosklearn,autokeras

6. autoviml

7. autoViz

8. hyperopt

8. sweetviz (EDA purpose)  - https://pypi.org/project/sweetviz/

9. pandasprofiling(display whole EDA) - https://pypi.org/project/pandas-profiling/

10. autokeras,AutoSklearn 

11. pycaret- https://pycaret.org/

12.Auto_Timeseries by auto_ts 

13.AutoNLP_Sentiment_Analysis by autoviml

14.automl lazypredict https://github.com/shankarpandala/lazypredict 

15.bamboolib or pandas-ui or pandas-summary or pandas_visual_analysis or Dtale(get code also) (python package for easy data exploration & transformation)  

   Automating EDA using Pandas Profiling, Sweetviz and Autoviz,vaex
   
   ExploriPy import EDA-https://analyticsindiamag.com/hands-on-tutorial-on-exploripy-effortless-target-based-eda-tool/

16.CUPY (array process parallel in gpu)  https://pypi.org/project/cupy/

17.Dabl has a built-in function that will automatically detect data types and quality issues and apply appropriate pre-processing to a dataset to prepare it for machine learning.  https://pypi.org/project/dabl/

18.dask (parallel comptataion)   https://docs.dask.org/en/latest/

19.dataprep (Understand your data with a few lines of code in seconds)

   data-preparation-tools - https://improvado.io/blog/data-preparation-tools

20.Dora library is another data analysis library designed to simplify exploratory data analysis. https://pypi.org/project/Dora/

21.FastAPI is a modern, fast (high-performance), web framework for building APIs. https://fastapi.tiangolo.com/

22.faster Hyper Parameter Tuning(sklearn-nature-inspired-algorithms) https://pypi.org/project/sklearn-nature-inspired-algorithms/

23.FlashText (A library faster than Regular Expressions for NLP tasks)  https://pypi.org/project/flashtext/

24.Guietta (tool that makes simple GUIs simple)  https://pypi.org/project/guietta/

   pandas-visual-analysis -https://analyticsindiamag.com/hands-on-guide-to-pandas-visual-analysis-way-to-speed-up-data-visualization/

25.hummingbird (make code fastly exexcute) https://pypi.org/project/Hummingbird/

26.memory-profiler (tell memory consumption line by line)  https://pypi.org/project/memory-profiler/

27.numexpr (incerease speed of execution of numpy)  https://github.com/pydata/numexpr

28.pandarallel  (simple and efficient tool to parallelize your pandas computation on all your CPUs) https://pypi.org/project/pandarallel/

29.PDFTableExtract(by PyPDF2)  https://github.com/ashima/pdf-table-extract

30.PyImpuyte(Python package that simplifies the task of imputing missing values in big datasets)  https://pypi.org/project/PyImpuyte/

31.libra(Automates the end-to-end machine learning process in just one line of code)  https://pypi.org/project/libra/

32.debug code by puyton -m pdp -c continue 

33.cURL (This is a useful tool for obtaining data from any server via a variety of protocols including HTTP.)
   https://stackabuse.com/using-curl-in-python-with-pycurl/

34.csvkit  https://pypi.org/project/csvkit/
 
35.IPython  IPython gives access to enhanced interactive python from the shell. 

36.pip install faker  (Create our own Dataset)  https://pypi.org/project/Faker/

37.Python debugger    %pdb

38.𝚟𝚘𝚒𝚕𝚊-From notebooks to standalone web applications and dashboards https://voila.readthedocs.io/en/stable/  https://github.com/voila-dashboards/voila

39.𝚝𝚜𝚕𝚎𝚊𝚛𝚗  for timeseries data   https://github.com/tslearn-team/tslearn

40.texthero text-based dataset in Pandas Dataframe quickly and effortlessly  https://github.com/jbesomi/texthero

41.𝚔𝚊𝚕𝚎𝚒𝚍𝚘(web-based visualization libraries like your Jupyter Notebook with zero dependencies)   https://pypi.org/project/kaleido/

42.Vaex- Reading And Processing Huge Datasets in seconds  https://github.com/vaexio/vaex

43.Uber’s Ludwig is an Open Source Framework for Low-Code Machine Learning  https://eng.uber.com/introducing-ludwig/

44.Google's TAPAS, a BERT-Based Model for Querying Tables Using Natural Language  https://github.com/google-research/tapas

45.RAPIDS  open GPU Data Science  https://rapids.ai/

46.pyforest Lazy-import of all popular Python Data Science libraries. Stop writing the same imports over and over again. https://pypi.org/project/pyforest/0.1.1/

47.Modin Get faster Pandas with Modin  https://github.com/modin-project/modin

48.Text2Code for Jupyter notebook  - https://github.com/deepklarity/jupyter-text2code , https://towardsdatascience.com/data-analysis-made-easy-text2code-for-jupyter-notebook-5380e89bb493

49.Openrefine Tool-For Data Preprocessing Without Code  https://analyticsindiamag.com/openrefine-tutorial-a-tool-for-data-preprocessing-without-code/

50.Microsoft Releases Latest Version Of DeepSpeed  deep learning optimisation library known as DeepSpeed- https://github.com/microsoft/DeepSpeed

https://analyticsindiamag.com/microsoft-releases-latest-version-of-deepspeed-its-python-library-for-deep-learning-optimisation/

51.4-pandas-tricks-https://towardsdatascience.com/4-pandas-tricks-that-most-people-dont-know-86a70a007993

52.tkinter to deploy machine learning model-https://analyticsindiamag.com/complete-tutorial-on-tkinter-to-deploy-machine-learning-model/

53.autoplotter is a python package for GUI based exploratory data analysis-https://github.com/ersaurabhverma/autoplotter

54.3 NLP Interpretability Tools For Debugging Language Models-https://www.topbots.com/nlp-interpretability-tools/

55.New Algorithm For Training Sparse Neural Networks (RigL)-https://analyticsindiamag.com/rigl-google-algorithm-neural-networks/

56.Read Data from pdf and Word-PyPDF2,PDFMiner,PDFQuery,tabula-py,pdflib for Python,PDFTables,PyFPDF2

   OpenCV to Extract Information From Table Images-https://analyticsindiamag.com/how-to-use-opencv-to-extract-information-from-table-images/

57.Text Annotation-https://towardsdatascience.com/tortus-e4002d95134b

58.GDMix, A Framework That Trains Efficient Personalisation Models - https://analyticsindiamag.com/linkedin-open-sources-gdmix-a-framework-that-trains-efficient-personalisation-models/

59.Learn Machine Learning Concepts Interactively-https://towardsdatascience.com/learn-machine-learning-concepts-interactively-6c3f64518da2

60.Folium, Python Library For Geographical Data Visualization-https://analyticsindiamag.com/hands-on-tutorial-on-folium-python-library-for-geographical-data-visualization/

61.GPU Technology Conference (GTC) Keynote Oct 2020-https://www.youtube.com/watch?v=Dw4oet5f0dI&list=PLZHnYvH1qtOYOfzAj7JZFwqtabM5XPku1

62.jiant nlp task-https://github.com/nyu-mll/jiant

63.painted your machine learning model-https://koaning.github.io/human-learn/

64.Vector AI-https://github.com/vector-ai/vectorai

***I will be so happy that this repository helps you. Thank you for reading.***


                                                        HAPPY LEARNING


