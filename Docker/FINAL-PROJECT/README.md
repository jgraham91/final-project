Workflow:

Data Understanding:
First, I would carefully analyze the data and understand its structure, distributions, and patterns. I would also look for any missing or incorrect data and determine how to deal with it.

Exploratory Data Analysis (EDA):
Next, I would perform EDA to gain insights into the data and identify any patterns or correlations. This could involve visualizations, statistical analysis, and feature engineering.

Data Preprocessing:
Based on my EDA findings, I would then preprocess the data to prepare it for modeling. This could involve scaling, normalization, encoding categorical features, handling missing values, and feature selection.

Model Selection:
Based on the problem at hand, I would choose appropriate models that could effectively solve the problem. Given that we are looking for a complex recommender engine that combines user ratings and content of the movies, I would consider models such as collaborative filtering, content-based filtering, and hybrid models.

Model Training and Evaluation:
I would then train and evaluate the models using appropriate metrics such as mean squared error, root mean squared error, and accuracy. This would involve splitting the data into training and testing sets, cross-validation, and hyperparameter tuning.

API Development:
Once we have the trained model, we can develop an API that takes the user_id as input and returns 3 recommended movies. The API can use the trained model to make predictions based on the user's interests and preferences.

Deployment:
Finally, we can deploy the API on a cloud platform such as Amazon Web Services or Google Cloud Platform. This would enable users to access the recommender engine through a web or mobile interface.