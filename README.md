# Bayes Classifier from scratch
I wrote this program as General as possible so you can use it for different datsets, I used 'iris' and also apply different PDFs, my program usses Gaussion PDF as the p(y|x = x1 ,..xk)

I examined Bayes classifier with covariance matrix and digonal matrix and proved thar Bayes classifier is equal to Naiive Bayes 

## Proof
The Gaussian function of Bayesian classifier is as follows:

![image](https://user-images.githubusercontent.com/47606879/147841528-1300a098-bd8d-433c-afa2-f4790b424e0d.png)

Digonal covariance Matrix:

![image](https://user-images.githubusercontent.com/47606879/147841604-911e4500-6cf6-471c-9db1-68aeb0fa5900.png)

![image](https://user-images.githubusercontent.com/47606879/147841577-c24b2851-dd18-4f7c-938c-f242aa378a89.png)

As you can see, the Bayesian probability distribution with the diagonal covariance matrix is multiplied by the distribution of individual properties, which is equivalent to Naiive Bayesian classifier.⬛

## Functions

### PDfBuilder
For the naiive bayes method, this function calculates the PDF as the sum of the natural PDF logarithms of each property. Also, this function returns the CDF value for calculating the Confidence matrix.
### The Naiive and Bayes
specify the PDF and CDF values for each class.

### Evaluate
This function takes the learning data and its label, the mean, the variance / covariance, the final labels, and the target class, which in this case is 'class', by taking the learning method, Naiive or Bayesian. For each data value, the PDF calculates the CDF based on the mean and variance / covariance and makes a prediction. It also calculates and returns Confusion and Confidence matrices in parallel, as well as accuracy.

### PreData
This function reads the data from the path and creates a DataFrame from it and also shuffles the data of this DataFrame 5 times.


### Parameters
This function calculates and returns the output of the PreData function, the training and test data, the labels for each category, the mean, the variance, and the covariance.

### K_Fold
This function has the same function as Parameters, except that it takes all the data and divides the data by the value of k, and puts the ith part as the test data and the rest as the training data.

### K_FoldCrossValidation
This function repeats the training and evaluation operations k times using the K_Fold function. The BayesM variable specifies the Bayesian type. If it is zero, Bayesian learning is done with covariance, and if it is another value, it uses variance according to the input learning method.


### Analysis
This function examines the learning method for training and test data and returns a list with len = 2  of the output of the Evaluate function. The first index of the output list is the result of the evaluation of test data and the second index is the result of educational data.


### MatPlot
This function plots the input matrix. This function was created to plot Confusion and Confidence diagrams


## Result + Confusion and Confidence matrices

### Bayes
- Iteration 1 on Test: Bayes: Accuracy is  97.73
- Iteration 1 on Train: Bayes: Accuracy is  99.06
- Iteration 2 on Test: Bayes: Accuracy is  100.00
- Iteration 2 on Train: Bayes: Accuracy is  98.11
- Bayes: Average Accuracy is  98.72

![Bayes Confusion Matrix](https://user-images.githubusercontent.com/47606879/147841447-ffe0a25a-004a-45bc-8115-b4d18c458869.png)
![Bayes Confidence Matrix](https://user-images.githubusercontent.com/47606879/147841449-d7887bc7-2730-4013-a5ad-cd3fb8c246f2.png)

4- fold Bayes: Average Accuracy is  96.53

![4 fold Bayes Confusion Matrix](https://user-images.githubusercontent.com/47606879/147841454-4bf2b39e-ed48-4e50-ad48-fc81bfbdc85d.png)

![4 fold Bayes Confidence Matrix](https://user-images.githubusercontent.com/47606879/147841452-f04d103b-4e96-4fe1-b9fa-9d6ac0778dd4.png)


### Bayes with diagonal covariance matrix

- Iteration 1 on Test: Bayes with diagonal covariance: Accuracy is  93.18
- Iteration 1 on Train: Bayes with diagonal covariance: Accuracy is  95.28
- Iteration 2 on Test: Bayes with diagonal covariance: Accuracy is  100.00
- Iteration 2 on Train: Bayes with diagonal covariance: Accuracy is  94.34
- Bayes with diagonal covariance: Average Accuracy is  95.70

![Bayes with diagonal covariance Confusion Matrix](https://user-images.githubusercontent.com/47606879/147841458-fa3c7f07-ee73-4ce9-83b7-e9f5d520c2df.png)
![Bayes with diagonal covariance Confidence Matrix](https://user-images.githubusercontent.com/47606879/147841459-bfa25a03-644a-4dd7-ba87-8b6f28732903.png)

4- fold Bayes with diagonal covariance: Average Accuracy is  95.83

![4 fold Bayes with diagonal covariance Confusion Matrix](https://user-images.githubusercontent.com/47606879/147841461-ca57e9c9-efe2-485e-a9ab-94b54bda3dfe.png)
![4 fold Bayes Confidence Matrix](https://user-images.githubusercontent.com/47606879/147841477-d56fff17-550f-43d5-8cfb-ae7ffeeb3db6.png)


### Naiive Bayes

- Iteration 1 on Test: Naiive Bayes: Accuracy is  93.18
- Iteration 1 on Train: Naiive Bayes: Accuracy is  96.23
- Iteration 2 on Test: Naiive Bayes: Accuracy is  95.45
- Iteration 2 on Train: Naiive Bayes: Accuracy is  97.17
- Naiive Bayes: Average Accuracy is  95.51

![Naiive Bayes Confusion Matrix](https://user-images.githubusercontent.com/47606879/147841471-0ab3c455-6367-4bc4-b530-0178b17ea10e.png)
![Naiive Bayes Confidence Matrix](https://user-images.githubusercontent.com/47606879/147841475-e15a7fc8-c96f-4a11-9920-77f0e2688ab5.png)


4-fold Naïve Bayes: Average Accuracy is  95.83

![4 fold Naiive Bayes Confusion Matrix](https://user-images.githubusercontent.com/47606879/147841480-5f8fc6bf-cde7-4ad4-9781-0c19e2b9a531.png)

![4 fold Naiive Bayes Confidence](https://user-images.githubusercontent.com/47606879/147841481-396e439a-4c09-4441-b9a8-14bd02b4698f.png)
