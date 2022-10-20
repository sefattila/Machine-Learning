import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing as pr
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 

veri=pd.read_csv('C:/Users/AttilA/Desktop/Odev_1/SydneyHousePrices.csv')

veri2=veri.drop(['Date','postalCode','Id'],axis=1)


suburb=veri2.iloc[:,0:1].values
propType=veri2.iloc[:,-1].values

le = pr.LabelEncoder()

suburb[:,0]=le.fit_transform(veri2.iloc[:,0])
propType[:]=le.fit_transform(veri2.iloc[:,5])



suburbdf=pd.DataFrame(data=suburb)

veri2=veri2.drop(['suburb'],axis=1)

veriSon=pd.concat([veri2,suburbdf],axis=1)

veriSon2=veriSon.rename(columns={0:"suburb"})
veriSon2=veriSon2.dropna(axis=0)


x=veriSon2.iloc[:,1:6]
y=veriSon2.iloc[:,0:1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

#linear
lr = LinearRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

plt.scatter(x[['car']], y.values, color='blue')
plt.plot(x_test[['car']], y_pred,color='red')
plt.show()

#polinom
poly_reg = pr.PolynomialFeatures()
x_poly=poly_reg.fit_transform(x.values)

x_train_poly, x_test_poly, y_train, y_test = train_test_split(x_poly,y.values,test_size=0.33,random_state=0)

lr2=LinearRegression()
lr2.fit(x_train_poly,y_train)
Y_pred_poly=lr2.predict(x_test_poly)


#StandardScale

sc = pr.StandardScaler()
x_train_olc=sc.fit_transform(x_train)
x_test_olc=sc.fit_transform(x_test)
y_train_olc=sc.fit_transform(y_train)
y_test_olc=sc.fit_transform(y_test)

'''
#SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_train_olc,y_train_olc)
y_pred_srv = svr_reg.predict(x_test_olc)
'''
#karar agacı

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(x_train,y_train)
y_pred_ka = dtr.predict(x_test)



