              # Neural Networks and Deep Machine Learning

<p align="center">
    <img width="500" height="300" src= "https://github.com/rloufoster/Neural_Network_Charity_Analysis/blob/main/Images/Neural_Network2.jpeg?raw=true">
</p> 


## Overview:

The purpose of this analysis was to explore and implement neural networks using TensorFlow in Python.  Neural networks is an advanced form of Machine Learning that can recognize patterns and features in the dataset. Neural networks are modeled after the human brain and contain layers of neurons that can perform individual computations.  A great example of a dDeep Learnning Neural Network would be image recognition.  The neural network will compute, connect, weigh and return an encoded categorical result to identify if the image represents a "dog" or a "cat".

#### AlphabetSoup

AlphabetSoup, a philanthropic foundation and is requesting for a data-driven solution that will help determine which organizations should be prioritized when it comes to the distribution of funds. Some past recipients have proven to not be the best stewards of AplphabetSoup funds, and moving forward, the foundation would prefer to be more selective in the distribution of funds in order to get the most impact for their donors. Beks, a data scientist for AlphabetSoup is tasked with analyzing the impact of each donation and vet the recipients to determine if the company's money will be used effectively, and what organizations may be "high risk". In order to accomplish this request, we are tasked with helping Beks create a binary classifier that will predict whether an organization will be successful with their funding. We utilize Deep Learning Neural Networks to evaluate the input data and produce clear decision making results.

### Deliverables:

 * Deliverable 1:  Preprocessing Data for a Neural Network Model
 * Deliverable 2:  Compile, Train, and Evaluate the Model
 * Deliverable 3:  Optimize the Model
 * Deliverable 4:  A Written Report on the Analysis 
 
 
### Resources:
 
 * Data Source:  charity_data.csv
 * Data Tools:  AlphabetSoupCharity_starter_code.ipynb
 * Software:  Python 3.7.6, Ananconda3 4., Jupyter Notebook and Pandas

  
The source data supplied to us was a csv file containing 34,000 organizations that are past recipients of Alphabet Soup funds. The dataset contained the following information:
 
  * **EIN and NAME** — Identification columns
  * **APPLICATION_TYPE** — Alphabet Soup application type
  * **AFFILIATION** — Affiliated sector of industry
  * **CLASSIFICATION** — Government organization classification
  * **USE_CASE** — Use case for funding
  * **ORGANIZATION** — Organization type
  * **STATUS** — Active status
  * **INCOME_AMT** — Income classification
  * **SPECIAL_CONSIDERATIONS** — Special consideration for application
  * **ASK_AMT** — Funding amount requested
  * **IS_SUCCESSFUL** — Was the money used effectively 
  
## Results:
  
### Deliverable 1:  Data Preprocessing

 * EIN and NAME columns added no value to the analysis and were dropped
 * APPLICAION_TYPE was binned and unique values with less than 500 records were classified as "Other"
 * IS_SUCCESSFUL was designated as the target variable
 * The remaining 43 variables were classified as features

### Deliverable 2:  Compiling, Training and Evaluating the Model

 * The initial model had a total of 5,981 parameters as a result of 43 inputs with 2 hidden layers and 1 output layer.

   - The first hidden layer had 43 inputs, 80 neurons and 80 bias terms.
   - The second hidden layer had 80 inputs (number of neurons from first hidden layer), 30 neurons and 30 bias terms.
   - The output layer had 30 inputs (number of neurons from the second hidden layer), 1 neuron, and 1 bias term.
   - Both the first and second hidden layers were activated using RELU - Rectified Linear Unit function. The output layer was activated          using the Sigmoid function.
   
 * The target performance for the accuracy rate is greater than 75%. The model that was created only achieved an accuracy rate of 72.33%

![Model1](https://github.com/rloufoster/Neural_Network_Charity_Analysis/blob/main/Images/Model1.png?raw=true)

### Deliverable 3:  Optimization Rounds

In order to attempt to improve accuracy rate, three optimization rounds were conducted that included adjustments such as adding/subtracting neurons and epochs.  The results showed no improvement.

![Optimization](https://github.com/rloufoster/Neural_Network_Charity_Analysis/blob/main/Images/OptimizationResults.png?raw=true)


 * **Optimization 1:
 
     - Binned INCOME_AMT column
     - Created 5,821 total parameters, a decrease of 160 from the original 
     - Accuracy improved 0.13% from 72.33% to 72.42%
     - Loss was reduced by 2.10% from 58.08% to 56.86%
     
 * **Optimization 2:
 
     - Removed ORGANIZATION column
     - Binned INCOME_AMT column
     - Removed SPECIAL_Considerations_Y column from features as it is redundant to SPECIAL_CONSIDERATIONS_N
     - Increased neurons to 100 for the first hidden layer and 50 for the second hidden layer
     - Accuracy decreased 0.11% from 72.33% to 72.24%
     - Loss increased by 1.75% from 58.08% to 59.10%
     
 * **Optimization 3:
 
     - Binned INCOME_AMT and AFFILIATION column
     - Removed SPECIAL_CONSIDERATION_Y column from features as it is redundant to SPECIAL_CONSIDERATIONS_N
     - Increased neurons to 125 for the first hidden layer and 50 for the second hidden layer
     - Created 11,101 total parameters, an increase of 5,120 from the optimal of 5,981
     - Accuracy increased 0.19% from 72.33% to 72.47%
     - Loss decreased by 1.82% from 58.08% to 57.02%
     
 ## Summary:
 
In summary, our model and various optimization rounds did not result in increased accuracy rate and reaching the 75% accuracy rate requested. With the adjustments of increasing the epochs, removing variables and/or increasing/decreasing the neurons, the changes were minimal and did not improve above 19 basis points. In reviewing other Machine Learning algorithms, the results did not prove to be any better. For example, Random Forest Classifier had a predictive accuracy rate of 70.80% which is a 2.11% decrease from the accuracy rate of the Deep Learning Neural Network model (72.33%).

Overall, Neural Networks are very intricate and would require experience through trial and error or many iterations to identify the perfect configuration to work with this dataset.

     
     
     -
 
