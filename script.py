#main
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#sklearn --preprocessing
from sklearn.model_selection import train_test_split
from sklearn_features.transformers import DataFrameSelector
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline,FeatureUnion
#sklearn --models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#sklearn  --metrics
from sklearn.metrics import f1_score,confusion_matrix,ConfusionMatrixDisplay
df=pd.read_csv('Hotel Reservations.csv')
df.drop(columns="Booking_ID",inplace=True)

x=df.drop(columns='booking_status')
y=df['booking_status']
x_train,x_test ,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,stratify=y,random_state=45)
num_cols=x_train.select_dtypes(include='number')
num_cols=list(num_cols.drop(columns=['arrival_month','arrival_date','arrival_year','required_car_parking_space','repeated_guest']).columns)
cat_columns=list(x_train.select_dtypes(include='object').columns)
ready_cols=['required_car_parking_space','repeated_guest']
num_pipe=Pipeline(steps=[
    ('selctor',DataFrameSelector(num_cols)),
    ('Scaling',StandardScaler())
])
cat_pipe=Pipeline(steps=[
    ('selector',DataFrameSelector(cat_columns)),
    ('OHE',OneHotEncoder()),
    ('scaling',StandardScaler(with_mean=False))  
])
ready_pipe=Pipeline(steps=[
    ('selector',DataFrameSelector(ready_cols)),
    ('Scaling',StandardScaler(with_mean=False))
])
#combine_pipes
all_pipe=FeatureUnion(transformer_list=[
    ('numerical',num_pipe),
    ('categorical',cat_pipe),
    ('ready',ready_pipe)

])
#apply
x_train_final=all_pipe.fit_transform(x_train)
x_test_final=all_pipe.transform(x_test)
with open('metrics.txt','w'):
    pass
def train_model(x_train,y_train):
    cls=RandomForestClassifier(n_estimators=300,max_depth=5)
    cls.fit(x_train,y_train)
    y_pred_train=cls.predict(x_train)
    y_pred_test=cls.predict(x_test_final)

    #calc f1_score
    Score_train=f1_score(y_train, y_pred_train,pos_label='Not_Canceled')
    Score_test=f1_score(y_test,y_pred_test,pos_label='Not_Canceled')
    #name of classifier
    cls_name=cls.__class__.__name__
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Canceled','Not_Canceled'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("confusion_matrix")
    plt.savefig(f'confusion_matrix.png')
    plt.close()
    with  open('metrics.txt','a') as f:
        f.write(f'{cls_name} \n')
        f.write(f'F1_score for Training is : {Score_train*100:.0f}%\n')
        f.write(f'F1_score for Testing is : {Score_test*100:.0f}%\n')
        f.write('****'*10+'\n')
    return True
train_model(x_train_final,y_train)