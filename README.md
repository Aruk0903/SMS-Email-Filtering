# SMS-Email-Filtering
# Spam classification with Naive Bayes and Support Vector Machines.
# Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline  
```
# Exploring the Dataset
```python
data = pd.read_csv('../input/spam.csv', encoding='latin-1')
data.head(n=10)
```
![image](https://user-images.githubusercontent.com/89111546/192168055-8d7dcb72-9c38-4da2-a0e4-99218783f1aa.png)

# Distribution spam/non-spam plots
```python
count_Class=pd.value_counts(data["v1"], sort= True)
count_Class.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
plt.show()
```
![image](https://user-images.githubusercontent.com/89111546/192168074-9952c4cc-a356-43c7-a786-df2b87ae206f.png)

```python
count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()
```
![image](https://user-images.githubusercontent.com/89111546/192168090-ac30c373-e8ab-46ce-ac0c-41d0e0e96686.png)

# Text Analytics

We want to find the frequencies of words in the spam and non-spam messages. The words of the messages will be model features.

We use the function Counter.
```python
count1 = Counter(" ".join(data[data['v1']=='ham']["v2"]).split()).most_common(20)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})
count2 = Counter(" ".join(data[data['v1']=='spam']["v2"]).split()).most_common(20)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})
```
```python
df1.plot.bar(legend = False)
y_pos = np.arange(len(df1["words in non-spam"]))
plt.xticks(y_pos, df1["words in non-spam"])
plt.title('More frequent words in non-spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()
```
![image](https://user-images.githubusercontent.com/89111546/192168118-b64a234c-b16d-47ce-a4e3-f19d5ee6e790.png)

```python
df2.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df2["words in spam"]))
plt.xticks(y_pos, df2["words in spam"])
plt.title('More frequent words in spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()
```
![image](https://user-images.githubusercontent.com/89111546/192168135-9d48147b-81b5-43a4-9d52-c2f92eef3b50.png)

We can see that the majority of frequent words in both classes are stop words such as 'to', 'a', 'or' and so on.

With stop words we refer to the most common words in a lenguage, there is no simgle, universal list of stop words.

# Feature engineering
Text preprocessing, tokenizing and filtering of stopwords are included in a high level component that is able to build a dictionary of features and transform documents to feature vectors.

We remove the stop words in order to improve the analytics

```python
f = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = f.fit_transform(data["v2"])
np.shape(X)
```
![image](https://user-images.githubusercontent.com/89111546/192168192-465df9ac-2786-484c-88b4-a7174561141c.png)

We have created more than 8400 new features. The new feature  j  in the row  i  is equal to 1 if the word  wj  appears in the text example  i . It is zero if not.

# Predictive Analysis
My goal is to predict if a new sms is spam or non-spam. I assume that is much worse misclassify non-spam than misclassify an spam. (I don't want to have false positives)

The reason is because I normally don't check the spam messages.
The two possible situations are:
New spam sms in my inbox. (False negative).
OUTCOME: I delete it.
New non-spam sms in my spam folder (False positive).
OUTCOME: I probably don't read it.

I prefer the first option!!!

First we transform the variable spam/non-spam into binary variable, then we split our data set in training set and test set.
```python
data["v1"]=data["v1"].map({'spam':1,'ham':0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.33, random_state=42)
print([np.shape(X_train), np.shape(X_test)])
```
![image](https://user-images.githubusercontent.com/89111546/192168298-14bc3464-8798-41b9-a129-10c3edb33262.png)

# Multinomial naive bayes classifier
We train different bayes models changing the regularization parameter  α .

We evaluate the accuracy, recall and precision of the model with the test set.
```python
list_alpha = np.arange(1/100000, 20, 0.11)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1 
```
Let's see the first 10 learning models and their metrics!
```python
matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns = 
             ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
models.head(n=10)
```
![image](https://user-images.githubusercontent.com/89111546/192168345-5e414885-891c-4388-ac1f-c0945b545839.png)

I select the model with the most test precision
```python
best_index = models['Test Precision'].idxmax()
models.iloc[best_index, :]
```
![image](https://user-images.githubusercontent.com/89111546/192168718-44359ed8-6520-4128-b088-aad27df083f4.png)

My best model does not produce any false positive, which is our goal.

Let's see if there is more than one model with 100% precision !
```python
models[models['Test Precision']==1].head(n=5)
```
![image](https://user-images.githubusercontent.com/89111546/192168375-78a70bdd-b3c3-43d6-9e60-5f0c78eedc6e.png)

Between these models with the highest possible precision, we are going to select which has more test accuracy.
```python
best_index = models[models['Test Precision']==1]['Test Accuracy'].idxmax()
bayes = naive_bayes.MultinomialNB(alpha=list_alpha[best_index])
bayes.fit(X_train, y_train)
models.iloc[best_index, :]
```
![image](https://user-images.githubusercontent.com/89111546/192168390-ce19cde0-16ee-4e49-ad51-07c48bb98918.png)

# Confusion matrix with naive bayes classifier
```python
m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])
```
![image](https://user-images.githubusercontent.com/89111546/192168412-8313efe6-f0e9-488c-a168-fb80b59c7d77.png)

We misclassify 56 spam messages as non-spam emails whereas we don't misclassify any non-spam message.


# Support Vector Machine
We are going to apply the same reasoning applying the support vector machine model with the gaussian kernel.
We train different models changing the regularization parameter C.
We evaluate the accuracy, recall and precision of the model with the test set.

```python
list_C = np.arange(500, 2000, 100) #100000
score_train = np.zeros(len(list_C))
score_test = np.zeros(len(list_C))
recall_test = np.zeros(len(list_C))
precision_test= np.zeros(len(list_C))
count = 0
for C in list_C:
    svc = svm.SVC(C=C)
    svc.fit(X_train, y_train)
    score_train[count] = svc.score(X_train, y_train)
    score_test[count]= svc.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, svc.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, svc.predict(X_test))
    count = count + 1 
```
Let's see the first 10 learning models and their metrics!
```python
matrix = np.matrix(np.c_[list_C, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns = 
             ['C', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
models.head(n=10)
```
![image](https://user-images.githubusercontent.com/89111546/192168441-abc19b46-87f0-47bb-b9f7-756a43bc581c.png)

I select the model with the most test precision
```python
best_index = models['Test Precision'].idxmax()
models.iloc[best_index, :]
```
![image](https://user-images.githubusercontent.com/89111546/192168451-0b9cca7a-17a8-41b8-b091-10ca66e56b7f.png)
My best model does not produce any false positive, which is our goal.

Let's see if there is more than one model with 100% precision !
```python
models[models['Test Precision']==1].head(n=5)
```
![image](https://user-images.githubusercontent.com/89111546/192168468-3f018730-18ce-4967-bd9e-5c81ce301ef0.png)

Between these models with the highest possible precision, we are going to selct which has more test accuracy.
```python
best_index = models[models['Test Precision']==1]['Test Accuracy'].idxmax()
svc = svm.SVC(C=list_C[best_index])
svc.fit(X_train, y_train)
models.iloc[best_index, :]
```
![image](https://user-images.githubusercontent.com/89111546/192168485-7900c773-4d4a-4200-8aaa-140703793604.png)

# Confusion matrix with support vector machine classifier.
```python
m_confusion_test = metrics.confusion_matrix(y_test, svc.predict(X_test))
pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])
```
![image](https://user-images.githubusercontent.com/89111546/192168498-7ab128e4-383c-4ab4-85ff-efe03a091a07.png)

We misclassify 31 spam as non-spam messages whereas we don't misclassify any non-spam message.

The best model I have found is support vector machine with 98.3% accuracy.

It classifies every non-spam message correctly (Model precision)

It classifies the 87.7% of spam messages correctly (Model recall)










