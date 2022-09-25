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

