from numpy import linalg as LA
from sklearn.model_selection import train_test_split
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import r_regression

titanic=pd.read_csv('TitanicSurvival.csv')
titanic.columns=['nomi', 'sopravvissuto','sesso','anni', 'classe_passe']


warnings.simplefilter(action='ignore', category=FutureWarning)

def normalizzation(array):
    array=array.to_numpy()
 
    stadev=array.std()
    stmean=array.mean()
    return (array-stmean)/stadev


X=titanic[titanic.anni.notna()]

X=X.replace(to_replace=['yes', 'no'],value=[1,0])
X=X.replace(to_replace=set(X['classe_passe']), value=[0,1,2])
X=X.replace(to_replace=['male','female'], value=[0,1])



X, X_test, y_train, y_test = train_test_split(X, X['sopravvissuto'], test_size=0.20, random_state=23)

#a=normalizzation(X['anni'])
#b=normalizzation(X['classe_passe'])

#X['anni']=a
#X['classe_passe']=b


matrix=X.iloc[:,2:]
cov_matrix=np.cov(matrix, rowvar=False)
eigenvalues, eigenvectors = LA.eig(cov_matrix)
print('eigen-val:',eigenvalues)
v1=eigenvalues[1]
v2=eigenvalues[2]

c1=eigenvectors[:,1]
c2=eigenvectors[:,2]

C=np.vstack((c1,c2))

final_data = C @ matrix.transpose() # Shape: (2, n_samples)

final_data=final_data.transpose()
x=final_data[0]
y=final_data[1]



plt.scatter(x[X.sopravvissuto == 1], y[X.sopravvissuto == 1], color='blue', label='sopravvissuto')
plt.scatter(x[X.sopravvissuto == 0], y[X.sopravvissuto == 0], color='red', label='morto')

#a_normalized_test=normalizzation(X_test['anni'])
#b_normalized_test=normalizzation(X_test['classe_passe'])

#X_test['anni']=a_normalized_test
#X_test['classe_passe']=b_normalized_test
matrix=X_test.iloc[:,2:]
final_test_data = C @ matrix.transpose() # Shape: (2, n_samples)


final_test_data=final_test_data.transpose()

x_t=final_test_data[0]
y_t=final_test_data[1]

x=x.to_numpy()
y=y.to_numpy()


distances=[]

for a,b,i in zip(x_t,y_t, range(len(x))):
    d=[]
    for a1, b1, j in zip(x[X.sopravvissuto == 1],y[X.sopravvissuto == 1], range(len(x[X.sopravvissuto==1]))):
        d.append([((a1-a)**2 + (b1-b)**2)**0.5,1])
    for a2, b2, j in zip (x[X.sopravvissuto == 0],y[X.sopravvissuto == 0], range(len(x[X.sopravvissuto==0]))):
        d.append([((a2-a)**2 + (b2-b)**2)**0.5, 0])
    distances.append(sorted(d, key=lambda x: x[0]))

voting=[]



y_test=y_test.to_numpy()

accuracy_result_k=[]
for k in range(100):

    
    voted=[]
    for distance in distances:
        yes=0 
        no=0
        for x,y in distance[:k]:
            if y == 1:
                yes+=1
            else:
                no+=1
        if yes>no:
            voted.append(1)
        else:
            voted.append(0)
    voted=np.array(voted)

   
    total=len(y_test)
    partial=0
    for vote,i in zip(voted, range(len(voted))):
        if vote == y_test[i]:
            partial+=1

    #print('accuracy: ', partial/total, 'partial: ',partial)
    accuracy_result_k.append(partial/total)
    
accuracy_result_k=np.array(accuracy_result_k)

print('best value: ', round(np.max(accuracy_result_k),2), 'whit k: ', np.where(accuracy_result_k == np.max(accuracy_result_k)))



plt.scatter(x_t,y_t, c='green', label='incerto')

plt.legend(loc='upper right')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()
