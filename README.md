# Cognifyz Technologies - Machine Learning Internship

There are four tasks given as per the internship task PDF document:

- ## Build a machine learning model to predict the aggregate rating of a restaurant based on other features.
- ## Create a restaurant recommendation system based on user preferences.
- ## Develop a machine learning model to classify restaurants based on their cuisines.
- ## Perform a geographical analysis of the restaurants in the dataset.

The objective is to build a Prediction model for Task 1, Recommendation model for Task 2,
Classification model for Task 3 and Clustering model for Task 4.

## Data Analysis, Visualization and Pre-Processing

The restaurant dataset was already provided, which contains of several features such as Restaurant ID, Name, City, Address, Longitude, Latitude, Cuisines, Average cost for two,
Price Range, Aggregate Rating, etc. I have used data pre processing techniques such as:

- Handling missing values in the dataset to ensure fairness in model training.
- Performed data analysis such as Mean, Standard Deviation, etc
- One-hot encoding “Cuisine” values from string to numerical attributes to train the model
- Converting “Cuisines” String attributes into numerical attributes using Text Vectorization for Recommendation model.
- Performed visualistic correlation between the parameters of the dataset using a correlation heatmap.

## Model Selection and Development

The models that are selected for implementation are:

1. ## Random Forest Regressor Model for Rating Prediction

- Split the Dataset into two parts:- Training the Model(80% dataset) and Testing the Model(20% dataset). Again we separate them into X_train, y_train for training our
model and X_test and y_test for the actual performance of the model.
- Load the Random Forest Regressor Model, which is an ensemble of multiple decision trees, used especially for making predictions on continuous variables, and fit
the X_train and y_train into the model.
- Compare the predictions of both Training Data and Testing Data and generated the R-Square scores of each.
- Built a predictive system, which takes the name of the restaurant to predict the aggregate rating of the restaurant.

2. ## Nearest Neighbours Algorithm with TF-IDF (Term Frequency-Inverse Document Frequency) for Restaurant Recommendation

- Create a TF-IDF vectorizer to transform the “Cuisines” column from string to numerical attributes and combined it with the other numerical features.
- Create a Nearest Neighbours Model using Cosine Similarity, which measures the angle between the vectors and fit the required features into the model.
- Built a recommendation system, which takes the important features such as Cuisines, Average cost for two and aggregate rating and recommends the top
restaurants based on the user preferences.

3. ## Random Forest Classifier Model for Restaurant Classification

- Split the Dataset into two parts:- Training the Model(80% dataset) and Testing the Model(20% dataset). Again we separate them into X_train, y_train for training our
model and X_test and y_test for the actual performance of the model.
- Load the Random Forest Classifier Model, which is an ensemble of multiple decision trees used for categorical variables, and fit the X_train and y_train into the
model.
- Compare the predictions of both Training Data and Testing Data and generate the accuracy scores of each.
- Built a classification-based system, which takes the name of the cuisine and classifies the restaurants based on the cuisine.

4. ## K-Means Clustering for Geographical Restaurant Analysis

- Extract the values of Geographical Longitude and Latitude columns from the dataset.
- Find the WCSS(Within Clusters Sum of Squares) value of the dataset.
- Plot an Elbow Graph to find the value of minimum number of clusters that we can group into.
- Train the Model according to the value of number of clusters we found from the elbow graph.
- Plot the clusters and their centroids using matplotlib.
- Calculate statistics such as average ratings and price ranges by City.

## Model Deployment and Hosting

Additionally, I've Built a Web-app using Streamlit for Recommendation System which contains a Title, Select box, Sub-header and Buttons, which
take relevant input data like:

• Cuisine
• Average Cost for two people
• Aggregate Rating

and recommends the top restaurants based on the user preferences. I've containerized it using Docker application and hosted it in the internet using Render.

Link:- https://recommendation-system-rlr6.onrender.com
