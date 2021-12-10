# Rahmadi-Ridwan-Portfolios


## Tensorflow-Portfolios:
Deep learning Tensorflow Keras projects in Image Classification, NLP and TSA

### [Rock, Paper, Scissors Hand Gestures Image Classification Project](https://github.com/rahmadiridwan/Tensorflow-Portfolio/blob/main/Batu_Gunting_Kertas.ipynb)
(https://github.com/rahmadiridwan/Tensorflow-Portfolio/blob/main/Batu_Gunting_Kertas.ipynb)

This is my first ever project using the Tensorflow Keras library as part of a machine learning course I took in Dicoding. The first task assigned in the course was to build a neural network using the Tensorflow Keras library to classify images of "rock, paper and scissors" hand gestures
* The raw dataset contains 2188 sample pictures of rock, paper and scissors hand gestures (of roughly equal amount for each)
* An ImageDataGenerator is deployed for augmentation purposes (to reduce bias in the training process)
* A sequential model consisting of 3 convolutional hidden layers (consisting of a 2 dimensional convolution layer and a max pooling layer) followed by a hidden dense layer (after flattening the output of the convolutional layers) is constructed to train the classifier. Dropout layers set at a rate of 20% are also applied alternately after layers to prevent overfitting.
* A callback API, ReduceLROnPlateau is also implemented before compiling the model to reduce the learning rate by a factor of 0.5 should the model 'plateaus' (or in other words, the learning rate stagnates)
* The model is compiled with the loss function and the optimizer set to be categorical crossentropy and adam respectively, whilst the metric evaluated being the accuracy
* The model is then fitted against the training along with the validation dataset for up to 15 epochs
* The model is shown to be effective in producing a training accuracy of 96.35% and a validation accuracy of 95.88% 
* A function to classify an uploaded image of a hand gesture to be either as rock, paper or scissors called image_predictor is defined and applied to test the trained model


### [NLP Text Classification of News Articles Project](https://github.com/rahmadiridwan/Tensorflow-Portfolio/blob/main/RahmadiRidwan_Proyek1NLP.ipynb)
(https://github.com/rahmadiridwan/Tensorflow-Portfolio/blob/main/RahmadiRidwan_Proyek1NLP.ipynb)

This is my first NLP project with Tensorflow Keras and my second Tensorflow project overall. This is also a project which is part of the machine learning course I took in Dicoding and this is the second task that I was assigned 
* The dataset for this project is the BBC News dataset which consists of 2225 samples
* The first step taken was the removal of stopwords from the dataset (in order that common words which are not good indicators to classify the text data are not weighted in the training process)
* The text data and its labels are split into seperate training and validation sets using the sklearn train_test_split library
* The contents of the training and validation sets are then tokenized (limited to its 5000 most frequent words in the text training and validation set which includes an 'out of vocabulary' index)
* Padding is also applied to the text training and validation sets to produce sequences of the tokenized words in numpy arrays of equal lengths
* The label traing and validation sets are also tokenized and padded
* A sequential model consisting of an Embedding layer (to group the text data as vector representations based on its inclinations), a Bidirectional LSTM layer and an output dense layer is constructed along with a dropout layer set at a rate of 50%
* The model is compiled with the loss function and the optimizer set to be categorical crossentropy and adam respectively, whilst the metric evaluated being the accuracy
* The model is then fitted against the training along with the validation dataset for up to 12 epochs
* The model is shown to be effective in producing a training accuracy of 99.83% and a validation accuracy of 95.28% 


### [Time Series Analysis on Weather Data](https://github.com/rahmadiridwan/Tensorflow-Portfolio/blob/main/Time_Series.ipynb)
https://github.com/rahmadiridwan/Tensorflow-Portfolio/blob/main/Time_Series.ipynb

This is my final project in the Dicoding machine learning course as well as my first time series analysis project that implements the Tensorflow Keras library where the task given is to train a recurrent neural network model that forecasts the weather (temperature) time series data and the mean absolute error of the regression model produced as the loss metric is kept at a minimum (under 10%)
* The dataset for this project is a record of weather (temperature) data for countries around the world, which consists of over 240000 samples
* The sample data is filtered to only include the climate data in China
* A graph plot of the date time and its respective temperature reading is then plotted using the matplotlib.pyplot library
* A function called windowed_dataset is created so the time series data can formatted to meet the requirement as input data to the neural network that is to be built
* The training and validation set is created by splitting the original filtered time series dataset by 80% and 20% respectively
* A sequential model consisting of two Bidirectional LSTM layers, two dense hidden layers and a dense output layer is built 
* The MAE (Mean Absolute Error) variable is defined as the difference between the maximum and the minimum average temperature
* The model is compiled with the loss function and the optimizer set to be huber and SGD respectively, whilst the metric evaluated being the accuracy
* The model is then fitted against the training along with the validation dataset for up to 60 epochs (with the application of a callback function)
* The model is shown to be effective in reducing the mean absolute error metric to 2.37 (10% its original value)



## NLP-with-Scikit-Learn
NLP Projects inspired by research papers implemented using the Python Scikit Learn library

### [Text Classification of News Articles using FCD and Naive-Baiyes](https://github.com/rahmadiridwan/NLP-with-Scikit-Learn/blob/main/MNB_FCD.ipynb)
https://github.com/rahmadiridwan/NLP-with-Scikit-Learn/blob/main/MNB_FCD.ipynb

This text mining project was inspired by a paper by Meng et al published in 2011 titled "A two-stage feature selection method for text categorization" (https://www.sciencedirect.com/science/article/pii/S089812211100616X) where a two stage feature selection method is implemented before applying support vector machine to classify text data. This project is my attempt to replicate the implementation of the first stage of the feature selection technique called FCD (Feature Contribution Degree) which is first proposed in this paper and to see how effective it is as a feature selection in its own. Hence, the second part of the feature selection stage (namely LSI) is not implemented and instead of the use of the SVM classifier, I will try to see how effective this feature selection technique is when applied together with the Multinomial Naive-Bayes classifier. (Note that this is also one of my very first projects in text mining using Python, when I tried to simply replicate and  maybe modify multiple feature selection techniques from text mining and NLP research papers as inspiration for my capstone project for my bachelor's degree in computing) 
* The text dataset provided for this project is an Indonesian news dataset consisting of 200 samples and 5 categories (with each category each consisting of 40 samples)
* At the text pre-processing stage, stopwords (in Indonesian) and punctuation marks are removed (using the nltk stopwords and the re libraries respectively) from the dataset and the text data is then stemmed and tokenized accordingly
* The dataset is then transormed by means of the binary countvectorizer into a matrix where the rows represent each sample, the columns represent each preprocessed word (term) in the entire corpus and the element of the matrix represents the binarized values of the feature terms in each document/row sample (0 if the term in not present in the document sample and 1 if it is present)
* A new matrix is then created consisting of the fcd values for each feature term in each news category
* The terms where a fcd value under 0 is present in any of the category are removed from the feature set and the tf-idf matrix of the new feature set with respect to each document (row sample) is then fed into the Multinomial Naive-Bayes algorithm with a training to testing data split of 70% and 30% respectively
* The classification accuracy for the model is 79.66%


### [Text Classification of News Articles using FS-CHICLUST and Naive-Baiyes](https://github.com/rahmadiridwan/NLP-with-Scikit-Learn/blob/main/MNB_FCD.ipynb)
https://github.com/rahmadiridwan/NLP-with-Scikit-Learn/blob/main/MNB_FCD.ipynb

This text mining project was inspired by a paper by Sarkar et al published in 2014 titled "A Novel Feature Selection Technique for Text Classification Using Naive Bayes" 
(https://www.researchgate.net/publication/285463155_A_Novel_Feature_Selection_Technique_for_Text_Classification_Using_Naive_Bayes) 
A two stage feature selection method called FS-CHICLUST is implemented before applying Multinomial Naive-Bayes to classify text data. This project is my attempt to emulate the implementation of FS-CHICLUST along with the Naive-Bayes classifier using the same Indonesian news articles dataset I used previously (when I attempted to implement FCD as the feature selection technique using the Naive-Bayes classifier). I did this project as my capstone project for my undergraduate dissertation

* The text dataset provided for this project is an Indonesian news dataset consisting of 200 samples and 5 categories (with each category each consisting of 40 samples)
* At the text pre-processing stage, stopwords (in Indonesian) and punctuation marks are removed (using the nltk stopwords and the re libraries respectively) from the dataset and the text data is then stemmed and tokenized accordingly
* The dataset is then transormed by means of the countvectorizer into a matrix where the rows represent each sample, the columns represent each preprocessed word (term) in the entire corpus and the element of the matrix represents the frequency of the feature terms present in each document/row sample 
* A new matrix is then created consisting of the modified chi-squared statistics values for each feature term in each news category
* The terms where the chi-square statistics value falls under the given threshold is present in any of the category are removed from the feature set and the tf-idf matrix of the new feature set with respect to each document (row sample) is then fed into the Multinomial Naive-Bayes algorithm with a training to testing data split of 70% and 30% respectively
* The classification accuracy for the model is 72.88% at the point of the iteration when the total number of clusters is equal to 960
