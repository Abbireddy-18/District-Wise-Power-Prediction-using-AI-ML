The Power Prediction project is a machine learning-based web application that forecasts regional power consumption and production. Below is the step-by-step process for setting up and using the application, along with the technical requirements.



### ***1. Technical Requirements***

### 

To run this project, you need Python 3.x installed on your system. The following libraries are required:



* Flask: Web framework for the backend.
* Pandas: For data manipulation and analysis.
* NumPy: For numerical computations.
* XGBoost: For advanced gradient-boosted machine learning models.
* Scikit-learn: For Random Forest and Linear Regression models.
* Werkzeug: For secure password hashing and utilities.





### ***2. How to Use the Project***



###### ***Step 1: Start the Application***

###### 

Run the Flask server with the following command:



**PowerShell**

python app.py

Open your browser and navigate to http://127.0.0.1:5000.



###### ***Step 2: Authentication***

###### 

Signup: Create a new account on the Signup page.

Login: Use your credentials to log in.



###### ***Step 3: Upload Data***



* Go to the Upload section.
* Upload a CSV file containing historical power data.
* Requirement: The CSV must have these columns: Date, State, District, Consumption\_MW, and Production\_MW.



###### ***Step 4: Train the Models***



* Once the data is uploaded and previewed, click the Train Model button.

The system will process the data and train three different models (Random Forest, XGBoost, and Linear Regression) for every district found in your dataset.



###### ***Step 5: Generate Predictions***



* Navigate to the Predict page.
* Select the State and District you want to analyze.
* Choose a specific Year (up to 2030) or select "All" to see a 5-year forecast.
* Select the Algorithm you want to use for the prediction.



###### ***Step 6: Analyze Results***



* Charts: View the dynamic charts comparing historical trends with future predictions.
* Accuracy: Compare the accuracy scores of the different algorithms to see which one performed best for that specific district.
* Suggestions: Read the AI-generated energy-saving suggestions based on the predicted consumption patterns.



