AWS Certified Machine Learning Specialty
========================================

### Goals
- Take AWS MLS exam before the end of 2022
- Gain a broad understanding of ML, techniques and algorithms
- Complete Andrew Ng course
- Complte Google Course (same as Justin)

## TODO

[ ] Read: https://developers.google.com/machine-learning/glossary/
[ ] Review https://experiments.withgoogle.com/collection/ai

### 1. Tutorial Dojo Test results

| Date       | Test              | Result |
|:-----------|:------------------|:-------|
| 2022-04-23 | Review Mode Set 1 | 47.69% |


### 2. AWS Products/Services


**Amazon Augmented AI**
- build the workflows that are required for human review
- get low-confidence results from Amazon Textract’s AnalyzeDocument API operation reviewed by humans
- Amazon A2I console


**Amazon Comprehend**
- uncover the insights and relationships in your unstructured data
- service identifies the language of the text; extracts key phrases, places, people, brands, or events; understands how positive or negative the text is; analyzes text using tokenization and parts of speech, and automatically organizes a collection of text files by topic.
Amazon Rekognition 
- image and video analysis
- automate your image and video moderation workflows



**AWS CloudTrail (with Sagemaker)**
- captures all SageMaker API calls
- exception to this is InvokeEndpoint

**Amazon CloudWatch API**


**Amazon DynamoDB Streams**
- can only be used for capturing streaming data from a DynamoDB table

**AWS Glue**
- crawlers

**AWS Glue DataBrew**

**Amazon Kinesis Data Analytics**
- HOTSPOTS function
- detects relatively dense regions in your data
- RANDOM_CUT_FOREST function 


**Amazon Kinesis Data Streams**

**Amazon Kinesis Data Streams APIs**
- help you manage many aspects of Kinesis Data Streams


**Amazon Kinesis Client Library (KCL)**
- helps you consume and process data from a Kinesis data stream

**Amazon Kinesis Producer Library (KPL)**
-  used for streaming data into Amazon Kinesis Data Streams and not for reading data from it



**Amazon SageMaker**

**Amazon SageMaker Neo**
- service primarily used to optimize machine learning models for inference on edge devices

**SageMaker Ground Truth**
- easy to efficiently perform highly accurate data labeling

**Amazon Textract**
- more fitted for extracting information from paper documents as it can automatically organize text data


### 3. Algorithms

Latent Dirichlet Allocation (LDA) algorithm
- unsupervised
- mainly used to discover a user-specified number of topics
- may uncover new topics/categories from a document by determining the occurrence of each word

Factorization machines (FM) algorithm
- supervised algorithm
- solving regerssion and classification tasks
- good when solving problems with known output values based on previous results

### 4. ML Concepts



### 5. Links

https://aws.amazon.com/about-aws/whats-new/2017/10/amazon-kinesis-analytics-can-now-pre-process-data-prior-to-running-sql-queries/
https://aws.amazon.com/about-aws/whats-new/2018/01/aws-kms-based-encryption-is-now-available-in-amazon-sagemaker-training-and-hosting/
https://aws.amazon.com/about-aws/whats-new/2019/09/amazon-sagemaker-now-supports-accelerated-training-with-new-smaller-amazon-fsx-for-lustre-file-systems/
https://aws.amazon.com/about-aws/whats-new/2020/05/amazon-forecast-now-supports-new-automated-data-imputation-options/
https://aws.amazon.com/athena/faqs/
https://aws.amazon.com/augmented-ai/
https://aws.amazon.com/blogs/aws/amazon-lex-build-conversational-voice-text-interfaces/
https://aws.amazon.com/blogs/aws/new-amazon-s3-storage-class-glacier-deep-archive/
https://aws.amazon.com/blogs/aws/new-serverless-streaming-etl-with-aws-glue/
https://aws.amazon.com/blogs/aws/new-vpc-endpoint-for-amazon-s3/
https://aws.amazon.com/blogs/big-data/analyze-data-in-amazon-dynamodb-using-amazon-sagemaker-for-real-time-prediction/
https://aws.amazon.com/blogs/big-data/analyzing-data-in-s3-using-amazon-athena/
https://aws.amazon.com/blogs/big-data/build-and-automate-a-serverless-data-lake-using-an-aws-glue-trigger-for-the-data-catalog-and-etl-jobs/
https://aws.amazon.com/blogs/big-data/building-a-multi-class-ml-model-with-amazon-machine-learning/
https://aws.amazon.com/blogs/big-data/loading-ongoing-data-lake-changes-with-aws-dms-and-aws-glue/
https://aws.amazon.com/blogs/big-data/preprocessing-data-in-amazon-kinesis-analytics-with-aws-lambda/
https://aws.amazon.com/blogs/big-data/processing-amazon-kinesis-stream-data-using-amazon-kcl-for-node-js/
https://aws.amazon.com/blogs/big-data/stream-data-to-an-http-endpoint-with-amazon-kinesis-data-firehose/
https://aws.amazon.com/blogs/big-data/test-data-quality-at-scale-with-deequ/
https://aws.amazon.com/blogs/big-data/top-10-performance-tuning-tips-for-amazon-athena/
https://aws.amazon.com/blogs/compute/using-aws-lambda-and-amazon-comprehend-for-sentiment-analysis/
https://aws.amazon.com/blogs/database/using-collaborative-filtering-on-yelp-data-to-build-a-recommendation-system-in-amazon-neptune/
https://aws.amazon.com/blogs/iot/sagemaker-object-detection-greengrass-part-2-of-3/
https://aws.amazon.com/blogs/machine-learning/amazon-rekognition-adds-support-for-six-new-content-moderation-categories/
https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-automatic-model-tuning-now-supports-random-search-and-hyperparameter-scaling/
https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-supports-knn-classification-and-regression/
https://aws.amazon.com/blogs/machine-learning/analyze-us-census-data-for-population-segmentation-using-amazon-sagemaker/
https://aws.amazon.com/blogs/machine-learning/build-a-custom-entity-recognizer-using-amazon-comprehend/
https://aws.amazon.com/blogs/machine-learning/build-a-model-to-predict-the-impact-of-weather-on-urban-air-quality-using-amazon-sagemaker/
https://aws.amazon.com/blogs/machine-learning/building-a-customized-recommender-system-in-amazon-sagemaker/
https://aws.amazon.com/blogs/machine-learning/create-accessible-training-with-initiafy-and-amazon-polly/
https://aws.amazon.com/blogs/machine-learning/enhanced-text-classification-and-word-vectors-using-amazon-sagemaker-blazingtext/
https://aws.amazon.com/blogs/machine-learning/ensure-consistency-in-data-processing-code-between-training-and-inference-in-amazon-sagemaker/
https://aws.amazon.com/blogs/machine-learning/fast-cnn-tuning-with-aws-gpu-instances-and-sigopt/
https://aws.amazon.com/blogs/machine-learning/identifying-worker-labeling-efficiency-using-amazon-sagemaker-ground-truth/
https://aws.amazon.com/blogs/machine-learning/introducing-gluon-an-easy-to-use-programming-interface-for-flexible-deep-learning/
https://aws.amazon.com/blogs/machine-learning/k-means-clustering-with-amazon-sagemaker/
https://aws.amazon.com/blogs/machine-learning/managing-missing-values-in-your-target-and-related-datasets-with-automated-imputation-support-in-amazon-forecast/
https://aws.amazon.com/blogs/machine-learning/now-available-in-amazo
https://aws.amazon.com/blogs/machine-learning/perform-a-large-scale-principal-component-analysis-faster-using-amazon-sagemaker/
https://aws.amazon.com/blogs/machine-learning/predicting-customer-churn-with-amazon-machine-learning/
https://aws.amazon.com/blogs/machine-learning/securing-amazon-sagemaker-studio-connectivity-using-a-private-vpc/
https://aws.amazon.com/blogs/machine-learning/semantic-segmentation-algorithm-is-now-available-in-amazon-sagemaker/
https://aws.amazon.com/blogs/machine-learning/speed-up-training-on-amazon-sagemaker-using-amazon-efs-or-amazon-fsx-for-lustre-file-systems/
https://aws.amazon.com/blogs/machine-learning/train-faster-more-flexible-models-with-amazon-sagemaker-linear-learner/
https://aws.amazon.com/blogs/machine-learning/use-amazon-cloudwatch-custom-metrics-for-real-time-monitoring-of-amazon-sagemaker-model-performance/
https://aws.amazon.com/blogs/machine-learning/use-the-amazon-sagemaker-local-mode-to-train-on-your-notebook-instance/
https://aws.amazon.com/blogs/machine-learning/use-the-built-in-amazon-sagemaker-random-cut-forest-algorithm-for-anomaly-detection/
https://aws.amazon.com/blogs/machine-learning/using-amazon-textract-with-amazon-augmented-ai-for-processing-critical-documents/
https://aws.amazon.com/blogs/machine-learning/using-pipe-input-mode-for-amazon-sagemaker-algorithms/
https://aws.amazon.com/blogs/opensource/why-use-docker-containers-for-machine-learning-development/
https://aws.amazon.com/cloudtrail/
https://aws.amazon.com/comprehend/
https://aws.amazon.com/comprehend/faqs/
https://aws.amazon.com/datapipeline/
https://aws.amazon.com/glue/
https://aws.amazon.com/kinesis/data-analytics/whitepaper-real-time-anomaly-detection/
https://aws.amazon.com/lex/
https://aws.amazon.com/premiumsupport/knowledge-center/start-glue-job-crawler-completes-lambda/
https://aws.amazon.com/rekognition/content-moderation/
https://aws.amazon.com/s3/faqs/
https://aws.amazon.com/sagemaker/groundtruth/faqs/
https://aws.amazon.com/transcribe/faqs/
https://aws.amazon.com/translate/faqs/
https://cran.r-project.org/web/packages/miceRanger/vignettes/miceAlgorithm.html
https://cs231n.github.io/neural-networks-3/#loss
https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/cloudwatch_concepts.html
https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Dashboards.html
https://docs.aws.amazon.com/AmazonS3/latest/dev/object-lifecycle-mgmt.html
https://docs.aws.amazon.com/comprehend/latest/dg/custom-entity-recognition.html
https://docs.aws.amazon.com/databrew/latest/dg/recipe-actions.ONE_HOT_ENCODING.html
https://docs.aws.amazon.com/deepcomposer/latest/devguide/deepcomposer-basic-concepts.html
https://docs.aws.amazon.com/dms/latest/userguide/Welcome.html
https://docs.aws.amazon.com/firehose/latest/dev/data-transformation.html#data-transformation-flow
https://docs.aws.amazon.com/firehose/latest/dev/what-is-this-service.html
https://docs.aws.amazon.com/glue/latest/dg/add-job-streaming.html
https://docs.aws.amazon.com/glue/latest/dg/add-job.html
https://docs.aws.amazon.com/glue/latest/dg/author-job.html
https://docs.aws.amazon.com/glue/latest/dg/populate-data-catalog.html
https://docs.aws.amazon.com/glue/latest/dg/vpc-endpoints-s3.html
https://docs.aws.amazon.com/kinesisanalytics/latest/sqlref/sqlrf-random-cut-forest.html
https://docs.aws.amazon.com/machine-learning/latest/dg/amazon-machine-learning-key-concepts.html
https://docs.aws.amazon.com/machine-learning/latest/dg/binary-classification.html
https://docs.aws.amazon.com/machine-learning/latest/dg/binary-model-insights.html
https://docs.aws.amazon.com/machine-learning/latest/dg/cross-validation.html
https://docs.aws.amazon.com/machine-learning/latest/dg/data-transformations-reference.html#normalization-transformation
https://docs.aws.amazon.com/machine-learning/latest/dg/improving-model-accuracy.html
https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html
https://docs.aws.amazon.com/machine-learning/latest/dg/multiclass-classification.html
https://docs.aws.amazon.com/machine-learning/latest/dg/regression-model-insights.html
https://docs.aws.amazon.com/machine-learning/latest/dg/regression.html
https://docs.aws.amazon.com/machine-learning/latest/dg/retraining-models-on-new-data.html
https://docs.aws.amazon.com/machine-learning/latest/dg/step-4-review-model-and-set-cutoff.html
https://docs.aws.amazon.com/machine-learning/latest/dg/training-parameters1.html
https://docs.aws.amazon.com/machine-learning/latest/dg/tutorial.html
https://docs.aws.amazon.com/machine-learning/latest/dg/types-of-ml-models.html
https://docs.aws.amazon.com/polly/latest/dg/managing-lexicons.html
https://docs.aws.amazon.com/quicksight/latest/user/scatter-plot.html
https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ContinuousParameterRange.html
https://docs.aws.amazon.com/sagemaker/latest/dg/algorithms-choose.html
https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html
https://docs.aws.amazon.com/sagemaker/latest/dg/best-practices.html
https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html
https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html
https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-transform.html#data-wrangler-transform-cat-encode
https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html
https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_hyperparameters.html
https://docs.aws.amazon.com/sagemaker/latest/dg/encryption-at-rest-nbi.html
https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-model-validation.html
https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html
https://docs.aws.amazon.com/sagemaker/latest/dg/how-pca-works.html
https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html
https://docs.aws.amazon.com/sagemaker/latest/dg/interface-vpc-endpoint.html
https://docs.aws.amazon.com/sagemaker/latest/dg/kNN_how-it-works.html
https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html
https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html
https://docs.aws.amazon.com/sagemaker/latest/dg/logging-using-cloudtrail.html
https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html
https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html
https://docs.aws.amazon.com/sagemaker/latest/dg/pca.html
https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html
https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html
https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-dg.pdf
https://docs.aws.amazon.com/sagemaker/latest/dg/semantic-segmentation.html
https://docs.aws.amazon.com/sagemaker/latest/dg/sms-automated-labeling.html
https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html
https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html
https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html
https://docs.aws.amazon.com/streams/latest/dev/shared-throughput-kcl-consumers.html
https://docs.aws.amazon.com/textract/latest/dg/a2i-textract.html
https://docs.aws.amazon.com/vpc/latest/userguide/vpc-endpoints.html
https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/evolve.html
https://dzone.com/articles/how-to-be-a-hero-with-powerful-parquet-google-and
https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst#learning-task-parameters
https://github.com/NVIDIA/nvidia-docker
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9056556
https://learningfromusersworkshop.github.io/papers/HyperTuner.pdf
https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
https://mste.illinois.edu/courses/ci330ms/youtsey/scatterinfo.html
https://people.eecs.berkeley.edu/~vipul_gupta/oversketch.pdf
https://sagemaker.readthedocs.io/en/stable/
https://scholar.smu.edu/cgi/viewcontent.cgi?article=1091&context=datasciencereview
https://tutorialsdojo.com/amazon-athena/
https://tutorialsdojo.com/amazon-cloudwatch/
https://tutorialsdojo.com/amazon-comprehend/
https://tutorialsdojo.com/amazon-fsx/
https://tutorialsdojo.com/amazon-kinesis/
https://tutorialsdojo.com/amazon-lex/
https://tutorialsdojo.com/amazon-polly/
https://tutorialsdojo.com/amazon-s3/
https://tutorialsdojo.com/amazon-sagemaker/
https://tutorialsdojo.com/amazon-transcribe/
https://tutorialsdojo.com/amazon-translate/
https://tutorialsdojo.com/amazon-vpc/
https://tutorialsdojo.com/aws-certified-machine-learning-specialty-exam-study-path/
https://tutorialsdojo.com/aws-cheat-sheets-aws-machine-learning-and-ai/
https://tutorialsdojo.com/aws-cloudtrail/
https://tutorialsdojo.com/aws-database-migration-service/
https://tutorialsdojo.com/aws-glue/
https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
https://www.mathsisfun.com/data/correlation.html
https://www.slideshare.net/AmazonWebServices/build-deep-learning-applications-using-mxnet-and-amazon-sagemaker-aim418-aws-reinvent-2018/26
https://www.youtube.com/watch?v=oIDHKeNxvQQ
https://www.youtube.com/watch?v=yqNBkFMnsL8

### A. Unsorted

Anomaly detection

labeled datasets

Linear Regression

Reinforcement Learning

Classification

model artifact

training data

target attribute

input data attributes

Binary Classification

Multicalass Classification

multinomial logistic regression

logistic regression

classes

cut-off number

classification problems

Amazon Athena

Apache ORC

Apache Parquet

AWS Glue



Data Pipeline



overfitting

spike
Multiclass classification
- labeling target data into specific category
- 

K-means algorithm
- unsupervised
- find discrete groupings within data
- k-parameter for groupings

Random Cut Forest (RCF)
- unsupervised
- detects anoalous data points

Linear Regression

time-series dataset

anomaly score

validation accuracy

t- distributed Stochastic Neighbor Embedding (t-SNE) 
- data pre-processor for data containing highly corrolated values

highly correlated features

scaling on the image data

deep learning neural network performance

data augmentation
- artificially create more training data from existing training data
- apply domain-specific techniques

Amazon Sagemaker Ground Truth

Single Shot Multibox Detector (SSD)

AWS IoT Greengrass lambda

multi-label classification

convolutional neural network (ResNet)

transfer learning

Apache MXNet RecordIO

hyperparameter 

gradient descent

colour scaling
Principal Component Analysis (PCA) algorithm
- unsupervised machine learning algorithm
- dimensionality reduction
- new set of features called components (composites of the original features)

one-hot encoding
- allows algorithms that expect continuous features to use categorical features

Pearson correlation coefficient
- measuring the statistical relationship between two continuous variables
de-noising 

feature extraction

exploratory data analysis tool

matrix multiplication 
- a compute-intensive operation used to process sparse or scattered data produced by the training model

Amazon OpenSearch



Amazon Transcribe
- converting speech to text quickly

Amazon Lex
- building chatbots
Amazon Polly 
- text-to-speech service
Content-based filtering algorithm

SageMaker API 
- requires access to public API endpoints to function

SageMaker Runtime

AWS PrivateLink

hyperparameter tuning job

XGBoost algorithm
- popular and efficient open-source implementation
- gradient boosted trees algorithm
gradient boosted trees algorithm
classification model

correlation of:
- eta
- alpha
- max_depth
- min_child_weight
Area Under the Curve (AUC)-hyperparameter combination
root mean square error (RMSE)-hyperparameter combination
- used to evaluate the performance of regression models and not classification models

AUC metric

t-Distributed Stochastic Neighbor Embedding (t-SNE)
- mainly used to reduce the dimensionality of a dataset
- can not visualize correlation with t-SNE
- non-linear dimensionality reduction algorithm

scatter plots
- show how much one variable is affected by another 

correlation
- relationship between two variables is called their correlation
positive correlation

negative correlation

histogram
- used for visualizing data distribution

neural network model

loss function
- evaluates how effective an algorithm is at modeling the data

loss function is oscillating
- likely due to learning rate being too high

loss function slowly decreases (not oscillates)
- likely due to learning rate being too low

Stochastic Gradient Descent (SGD) algorithm

learning rate
- affects the speed at which the algorithm converges

batch sizes
- smaller batch sizes would actually slow down the learning process

Min-max scaling 
- normalizes data within a fixed range

Random cropping 
- data augmentation technique
- will not reducec dimensionality
AWS Glue
- serverless data integration service
-  discover, prepare, and combine data for analytics, machine learning, and application development


Apache Hive

Amazon EMR

AWS Glue Data Catalog

AWS Batch job

supervised model

target label
objective hyperparameter:
- multi:softmax
- feature_dim

Amazon SageMaker XGBoost algorithm

K-means algorithm
- unsupervised
- finds discrete groupings 
- number of groupings == k-parameter
- not suitable for labeled dataset problems

k-parameter 

Amazon SageMaker Neural Topic Model (NTM)
- topic modeling algortihm
- mainly used to organize corpus of documents into topics

Keras Convolutional Neural Network (CNN)
- mainly for image data problems



Random Cut Forest (RCF) algorithm

K-means algorithm

ordinal features

nominal features



Feature Scaling 
- type of feature engineering 
- used for normalizing features within a dataset in a fixed range

Dimensionality Reduction
- reduce the number of highly correlated features

Amazon FSx For Lustre

Confusion matrix 
- used for evaluating the model’s performance

Correlation matrix 
- shows the correlation coefficient between variables

Residual plots 

Root Mean Square Error (RMSE) 
- used for measuring the accuracy of the ML model
- smaller the value of the RMSE, the better is the predictive accuracy of the model

residuals 
- difference between the true target and the predicted target
- portion of the target that the model is unable to predict
- positive residual indicates that the model is underestimating
- negative residual indicates an overestimation
ground truth

Multiple Imputations by Chained Equations (MICE) 
- robust, informative method of dealing with missing data

Amazon Forecast
- Middle filling
- Back filling
- Future filling

convergence

Amazon SageMaker DeepAR 

SageMaker BlazingText algorithm
- natural language processing (NLP) and Text classification tasks

ML storage volume

Box plot 

Confusion matrix 
- tool for visualizing the performance of a multiclass model
- all possible combinations of correct and incorrect predictions

Correlation matrix 
- how close the predicted values are from true values

Scatterplot 

binary classification

macro average

weighted average

F1-measure

multiclass classifier

overfitting issue

validation loss is much greater than the training loss

saddle point

epoch number

dropout rate at the hidden layer

sampling noise





k-NN binary classification model
- most k-NN models use the Euclidean distance to measure how similar the target data is to a specific class

low dimensional dataset

Apache Kafka

Kafka Connect Amazon S3 sink connector

Kinesis Data Firehose
- does not support converting CSV files into Apache Parquet
- supports JSON

Amazon EMR cluster 

Apache Spark Structured Streaming

Amazon Managed Streaming for Apache Kafka (Amazon MSK)

Semantic segmentation 
- fine-grained, pixel-level approach to developing computer vision applications
- segmentation mask

Object2Vec 
- a Natural Language Processing algorithm
- does not work on images

Image classification 
- classifies an object in an image

Object detection 
- like image classification, but it uses a bounding box to identify objects in a picture

evaluation score

mean absolute percentage error (MAPE)

k-fold validation

cross-validation
- technique for evaluating ML models by training several ML models on subsets of the available input data and evaluating them on the complementary subset of the data
Amazon Mechanical Turk
- crowdsourcing marketplace
- outsource repetative tasks

Levenshtein distance
- edit distance
- closeness of two strings

Word2vec



Amazon Textract

Neural Machine Translation (MT)

Amazon SageMaker Neural Topic Model
- used for classifying or summarizing documents based on the topics detected in a corpus

training error rate

L2 regularization 
- results in smaller overall weight values and stabilizes the weights when there is a high correlation between the input features 

decrease n-grams size

 L1 regularization 
 - has the effect of reducing the number of features used in the model by pushing to zero the weights of features that would otherwise have small weights

 Regularization 

Residual Network (ResNet) model

NVIDIA Deep Learning AMI

NVIDIA Container Toolkit 

Linear Learner model

protobuf RecordIO format
- converts each observation in the dataset into a binary representation as a set of 4-byte floats

Amazon SageMaker
- Pipe mode
- File mode

Custom Entity Recognition
- enabling you to identify new entity types not supported as one of the preset generic entity types

regularization

drift

TFRecord 

HDF5 

TensorFlow Docker container

MXNet containers

Amazon Redshift
- specifically used for data warehousing where large data workloads are required

Kinesis Data Analytics

Inference code

data lake 
- centralized repository that allows you to store all your structured and unstructured data at any scale

Amazon Athena 
- interactive query service

VPC endpoint
- virtual device
- enables private connections between your VPC and supported AWS services and VPC endpoint services powered by AWS PrivateLink
- does not require an Internet gateway, virtual private gateway, NAT device, VPN connection, or AWS Direct Connect connection
- instances in your VPC do not require public IP addresses to communicate with resources in the service
- horizontally scaled, redundant, and highly available VPC components
- allow communication between instances in your VPC and services without imposing availability risks
- does not require a security group or a Network Access Control List (NACL)

AWS PrivateLink 
- is a technology that enables you to privately access services by using private IP addresses

Amazon CloudWatch (with Amazon SageMaker)
- statistics kept for 15-months
- metric represents a time-ordered set of data points that are published to CloudWatch
- log is a generated data tied to a specific event that happens over time

AWS Data Pipeline 
- suitable for running scheduled tasks
- supports the JDBC database, Amazon RDS, and Amazon Redshift
- does not support external MySQL databases
- If you need to replicate a database and also want to load on-going changes, use AWS DMS

AWS Database Migration Service (AWS DMS)
- cloud service that makes it easy to migrate relational databases, data warehouses, NoSQL databases, and other types of data stores

change data capture (CDC) data

Specificity
- metric needed to plot the ROC curve

Root-mean-square error (RMSE)
- not suitable for evaluating classification models
- performance metric is mostly used for regression models

Area Under the ROC Curve (AUC)
- Area Under the (Receiver Operating Characteristic) Curve
- industry-standard accuracy metric for binary classification models
- measures the ability of the model to predict a higher score for positive examples as compared to negative examples
- independent of the score cut-off
- can get a sense of the prediction accuracy of your model from the AUC metric without picking a threshold
- graphical plot that shows the diagnostic ability of a binary classifer system as its discrimination threshold is varied
- AUC metric returns a decimal value from 0 to 1 (1== accurate; 0.5 == no better than random; 0 is unusual -- usually indicate data issue)

Mean Absolute Percentage Error(MAPE)
- used to evaluate regression models
- percentage of errors between the paired observations expressing the same phenomenon

score threshold
- used to fine-tune ML model performance metrics
- changes level of confidence that model must have in a prediction before it considers the prediction to be positive
- changes number of false negatives and falue positive being tolerated
Amazon SageMaker automatic model tuning
- ContinuousParameterRanges
- Logarithmic scaling works only for ranges that have only values greater than 0
- Linear learning range uses values between .0001 and 1.0
- searching on a linear scale would, on average, devote 90 percent of your training budget to only the values between .1 and 1.0
- If you specify 0 as the minimum hyperparameter value in an Auto setting, Amazon SageMaker will never choose to use logarithmic scaling. 

uniform distribution

 

Amazon Kinesis Data Firehose 
-  fully managed service for delivering real-time streaming data to destinations such as Amazon Simple Storage Service (Amazon S3), Amazon Redshift, Amazon Elasticsearch Service (Amazon ES), Splunk, and any custom HTTP endpoint or HTTP endpoints owned by supported third-party service providers
- data transformation buffers incoming data at up to 3 MB by default
- to adjust the buffering size, use the  ProcessingConfiguration API with the ProcessorParameter called BufferSizeInMBs

Kinesis Data Streams 
- can’t be used to transform data on the fly and store the output data to Amazon S3

AWS Batch 
- service used to efficiently manage the necessary compute resources for batch processing jobs
- service can’t be used for ingesting streaming data

Amazon SageMaker multi-model endpoints
-  provide a scalable and cost-effective solution to deploying large numbers of models

Amazon SageMaker linear learner algorithm
- supports three data channels
	- train
	- validation
	- test
Amazon Kinesis Analytics
- transform data before it is processed by your SQL code
- provides Lambda blueprints for common use cases

viseme Speech Mark
- used to synchronize speech with facial animation (lip-syncing) 
- highlight written words as they’re spoken

Speech Synthesis Markup Language (SSML) 
- pronunciation tag
- emphasis tag 

Amazon Polly
- pronunciation lexicons
- XML file
- lexicons can be stored in an AWS region
- dones not support SSML tags

Factorization Machines

binary_classifier

Logistic Regression 
- predicts a binary output such as “0” or “1”.

linear regression
- predicts a numeric value
confusion matrix
- true negatives (TN)
- true positives (TP)
- false negatives (FN)
- false positives (FP)
- model precision
- model accuracy

ORC file format
- support predicate pushdown (aka. predicate filtering)

model fit
- important for knowing root cause of poor model accuracy
- linear function trying to fit a parabolic function => a characteristic of an underfitting model
- underfitting model -> model is too simple to recognize the variations in the target function

Amazon Neptune
Collaborative filtering algorithm
- based on (user, item, rating) tuples
- Compared to content-based filtering, collaborative filtering provides better results for diversity (how dissimilar recommended items are); serendipity (a measure of how surprising the successful or relevant recommendations are); and novelty (how unknown recommended items are to a user)
- computationally expensive and more complex and costly to implement and manage than content-based filtering
- has a cold start problem

Latent Dirichlet Allocation (LDA) algorithm

Random Cut Forest (RCF) algorithm
- algorithm is only suitable for detecting anomalous data points within a data set

Recurrent Neural Nets (RNNs)

Transformer-based models

self-attention

Label encoding
- convert categorical data into integer labels (e.g. 0,1,2,3,4)

Target encoding
- replacing categorical variables with just one new numerical variable and replacing each category of the categorical variable with its corresponding probability of the target

Tokenization
- commonly used in Natural Language Processing (NLP) where you split a string into a list of words

Error, Trend, Seasonality (ETS) forecasts 

Autoregressive Integrated Moving Average (ARIMA) 

Amazon SageMaker DeepAR 
- forecasting algorithm 
- supervised
- uses recurrent neural networks (RNN)
- can provide better forecast accuracies compared to classical forecasting techniques
- requires little or no historical data
- neural network-based algortihm
- good for cold start problems

Exponential Smoothing (ES)



