import numpy as np
from math import sin
f=lambda x:1/(1+np.exp(-x))
def sigmoid(x,deriv=False):
 if(deriv==True):
   return f(x)*(1-f(x))
 return f(x)
def empty_act(x,deriv=False):
  if deriv==True:
    return 1
  return x   
f1=np.vectorize(empty_act)
def ReLu(x, deriv=False):
  if deriv==True:
   return sigmoid(x)
  return np.log(1+np.exp(x))   
def gr(X,Y):
  return X-Y
def MSE(X,Y):
  return np.mean((X-Y)**2)
def neirons_weights(a):
  n=[]
  for i in range(len(a)-1):
    n.append((a[i+1],a[i]))
  return(n)

class NN:
 def __init__(self,l,neirons,activations,gradient):
   #���������� �������� ���������� ������ ���������
   #l-���-�� �����(��� ������� � ��������),neirons-������������������ ���-�� �������� � ������ ����(������� ������ � ���������),activations- ������ ������� ���������(True-�����������),gradient-����������� ������ ���������(������).
   self.size=neirons_weights(neirons) # ������ �������� ��������� ������ �����
   self.activations=activations
   self.W=[] # ������ ������ �����
   self.B=[] # ������ �������� �������
   self.l=l
   for i in range(self.l):
     self.W.append( np.random.rand(*self.size[i]))
     self.B.append( np.random.rand(self.size[i][0],1))
   # ���� �������������� ������� ��������� ���� � ������  
   self.gradient=gradient

 def feedforward(self,x): # ������ ������
   self.input=x # ������� ������ 
   self.layer =[self.input] # ������ ��������� �����
   self.derivative=[] # ������ ����������� ��������� �����
   for i in range(self.l):
     self.layer.append(self.activations[i](np.dot(self.W[i],self.layer[i])+self.B[i])) # �������� ������ ������ �������� ������� an+1=a(Wn+1 X an + b)
     self.derivative.append(self.activations[i](np.dot(self.W[i],self.layer[i])+self.B[i],True)) # �������� ��������-�����������
   return self.layer[-1]


 def backprop(self,y): # �������� ������
   self.y=y # �������� ������
   J=self.gradient(self.layer[-1],self.y)*self.derivative[-1] # �������� ����������(�������) �� ����������� ��������� ���������  (1��� ��������� ��������� ��������������� ������)
   self.Delta=[J] # self.Delta-������ �������� ������ �����
   for i in range(self.l-1):
      self.Delta.append(self.W[self.l-1-i].T.dot(self.Delta[i])*self.derivative[self.l-2-i])
      #(2��� ��������� ��������������� ������) ((Wn+1)^T X dn+1)*dan/dz
   self.Delta=self.Delta[::-1]  # ���������� ������
   for i in range(self.l):
     Delta_W=(np.dot(self.Delta[i], self.layer[i].reshape(1, len(self.layer[i]))))*self.learning_rate 
     self.W[i]-=Delta_W # (3���) ��������������� ������ �� ����� � ���������
     self.B[i]-=self.Delta[i]*self.learning_rate #(4���)
 def Train(self,X,Y,learning_rate,epochs): # ���������� ��� ������� �������� 
   self.learning_rate=learning_rate
   self.epochs=epochs
   k=len(X)
   for i in range(self.epochs):
     a=[]
     b=[]
     for j in range(k):
       self.feedforward(X[j])
       self.backprop(Y[j])
       a.append(self.layer[-1])
       b.append(Y[j])
     print(MSE(np.array(a),np.array(b)))

















X=[4*i/100 for i in range(1,100) ]
Y=[np.array(i**2+sin(5*i)).reshape(1,1) for i in X ] # ����� � � Y ��������� ���������� f(x) �� ���������� �� 1 �� 4, ����� ,��������, f(x)=x**2+sin(x) 
X=[np.array(i).reshape(1,1) for i in X]
Aprochsin=NN(3,[1,5,5,1],[sigmoid,sigmoid,ReLu],gr) # ����� � �������� ����
Aprochsin.Train(X,Y,0.05,1000) # ����� ���� ��������
X=[4*i/100 for i in range(1,100) ]
Y=[(i,*Aprochsin.feedforward(np.array(i).reshape(1,1))[0]) for i in X]
print(Y) # ����� ����� ������� 100 ����� �� 0 �� 4 (x_i,y_i)
