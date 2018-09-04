# Data_Science_NYC_Fare

In this project, we predict the fare amount (inclusive of tolls) for a taxi ride in New York City given 
the pickup and dropoff locations. This is a featured competition problem in Kaggle.
https://www.kaggle.com/c/new-york-city-taxi-fare-prediction

# NYC_EDA.ipynb
- In this file we do exploratory data analysis of the dataset which has 55 million rows.
- We also run a basic model (avg. rate per km), Linear Regressor and DNN regressor on the data set.
- We divide the data into train, validation and test data set.
- We train using approx. 4 million rows.
- Report RMSE on the validation set.

# NYC_DNNLinear.ipynb
- RMSE on validation data on previous EDA run on different models is around 9
- We want to do better than this
- We engineer new features using feature cross 
- We feature cross latitude and longitude data to divide NYC into a grid and capture start and end point better and traffic codnitions.
- We feature cross day and time as well to capture taxi ride demand and traffic condition
- Feature crosses are made using Tensorflow 
- Run DNN Linear regressor (a canned Tensorflow model) using Datalab
- Monitor progress using Tensor Board
- Hyperparameter tunning by running different hidden layer sizes, 64,64,64,8 works best 
- RMSE on validation dataset is now down to 4.75

# NYC_CMLE_cloudrun.ipynb
- We can leverage the full power of Google Cloud ML Engine, by running our model as a job on CMLE
- This file sets up the environment to submit job to CMLE
- Estimator code has been packaged into model.py and task.py which are submitted to the CMLE
- RMSE on validation dataset is again aroun 4.75 as expected.

# TensorBoard
- Progress in tensorflow can be monitored using tensorboard
- Valid RMSE vs Step size shows RMSE going down as training progresses
