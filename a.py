from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
def CostFunction(Neurons4):
    Expected=np.array((0,0,0,0,0,0,0,1,0,0),dtype='float')
    CostL=[]
    count=0
    while count!=10:
        print(Neurons4[count])
        Cv=((Neurons4[count]-Expected[count])*(Neurons4[count]-Expected[count]))
        CostL.append(Cv)
        count+=1
    costFunction=sum(CostL)
    print("\n",costFunction,"\n")

    action='Unkown'
    if Neurons4[7]>Expected[7]:
        action="increase"
    if(Neurons4[7]<Neurons4[0] or Neurons4[7]<Neurons4[1] or Neurons4[7]<Neurons4[2] or Neurons4[7]<Neurons4[3] or Neurons4[7]<Neurons4[4] or Neurons4[7]<Neurons4[5] or Neurons4[7]<Neurons4[6] or Neurons4[7]<Neurons4[8] or Neurons4[7]<Neurons4[9]):
        action="Decrease"
    print(action)



def Predictions(Neurons4):
    #print(Neurons4)
    prediction=np.argmax(Neurons4)
    print("prediction:",prediction)
    CostFunction(Neurons4)
def Simgmoid(x):
    Sg=1/(1+math.exp(-x))
    return Sg
def OutputLayer(Neurons3):
    global OWeights
    OBs=[]
    Neurons4=[]
    OWeights1=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    OWeights2=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    OWeights3=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    OWeights4=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    OWeights5=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    OWeights6=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    OWeights7=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    OWeights8=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    OWeights9=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    OWeights10=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    x1=np.dot(Neurons3,OWeights1)
    x2=np.dot(Neurons3,OWeights2)
    x3=np.dot(Neurons3,OWeights3)
    x4=np.dot(Neurons3,OWeights4)
    x5=np.dot(Neurons3,OWeights5)
    x6=np.dot(Neurons3,OWeights6)
    x7=np.dot(Neurons3,OWeights7)
    x8=np.dot(Neurons3,OWeights8)
    x9=np.dot(Neurons3,OWeights9)
    x10=np.dot(Neurons3,OWeights10)
    Sg1=Simgmoid(x1)
    Sg2=Simgmoid(x2)
    Sg3=Simgmoid(x3)
    Sg4=Simgmoid(x4)
    Sg5=Simgmoid(x5)
    Sg6=Simgmoid(x6)
    Sg7=Simgmoid(x7)
    Sg8=Simgmoid(x8)
    Sg9=Simgmoid(x9)
    Sg10=Simgmoid(x10)
    Neurons4.append(Sg1)
    Neurons4.append(Sg2)
    Neurons4.append(Sg3)
    Neurons4.append(Sg4)
    Neurons4.append(Sg5)
    Neurons4.append(Sg6)
    Neurons4.append(Sg7)
    Neurons4.append(Sg8)
    Neurons4.append(Sg9)
    Neurons4.append(Sg10)
    Neurons4=np.array(Neurons4)
    print(len(Neurons4))
    #print(Neurons4)
    #print(OWeights)
    Predictions(Neurons4)
def HiddenLayer2(Neurons2):
    global HL2Weights
    HL2Bs=0
    Neurons3=[]
    HL2Weights1=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    HL2Weights2=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    HL2Weights3=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    HL2Weights4=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    HL2Weights5=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    HL2Weights6=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    HL2Weights7=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    HL2Weights8=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    HL2Weights9=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    HL2Weights10=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    HL2Weights11=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    HL2Weights12=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    x1=np.dot(Neurons2,HL2Weights1)
    x2=np.dot(Neurons2,HL2Weights2)
    x3=np.dot(Neurons2,HL2Weights3)
    x4=np.dot(Neurons2,HL2Weights4)
    x5=np.dot(Neurons2,HL2Weights5)
    x6=np.dot(Neurons2,HL2Weights6)
    x7=np.dot(Neurons2,HL2Weights7)
    x8=np.dot(Neurons2,HL2Weights8)
    x9=np.dot(Neurons2,HL2Weights9)
    x10=np.dot(Neurons2,HL2Weights10)
    x11=np.dot(Neurons2,HL2Weights11)
    x12=np.dot(Neurons2,HL2Weights12)
    Sg1=Simgmoid(x1)
    Sg2=Simgmoid(x2)
    Sg3=Simgmoid(x3)
    Sg4=Simgmoid(x4)
    Sg5=Simgmoid(x5)
    Sg6=Simgmoid(x6)
    Sg7=Simgmoid(x7)
    Sg8=Simgmoid(x8)
    Sg9=Simgmoid(x9)
    Sg10=Simgmoid(x10)
    Sg11=Simgmoid(x11)
    Sg12=Simgmoid(x12)
    Neurons3.append(Sg1)
    Neurons3.append(Sg2)
    Neurons3.append(Sg3)
    Neurons3.append(Sg4)
    Neurons3.append(Sg5)
    Neurons3.append(Sg6)
    Neurons3.append(Sg7)
    Neurons3.append(Sg8)
    Neurons3.append(Sg9)
    Neurons3.append(Sg10)
    Neurons3.append(Sg11)
    Neurons3.append(Sg12)
    Neurons3=np.array(Neurons3)
    print(len(Neurons3))
    #print(Neurons3)
    #print(HL2Weights)
    OutputLayer(Neurons3)
def HiddenLayer1(Neurons1):
    HL1Bs=0
    Neurons2=[]
    HL1Weights1=np.random.uniform(low=-.5000000,high=.75000000,size=784)
    HL1Weights2=np.random.uniform(low=-.5000000,high=.75000000,size=784)
    HL1Weights3=np.random.uniform(low=-.5000000,high=.75000000,size=784)
    HL1Weights4=np.random.uniform(low=-.5000000,high=.75000000,size=784)
    HL1Weights5=np.random.uniform(low=-.5000000,high=.75000000,size=784)
    HL1Weights6=np.random.uniform(low=-.5000000,high=.75000000,size=784)
    HL1Weights7=np.random.uniform(low=-.5000000,high=.75000000,size=784)
    HL1Weights8=np.random.uniform(low=-.5000000,high=.75000000,size=784)
    HL1Weights9=np.random.uniform(low=-.5000000,high=.75000000,size=784)
    HL1Weights10=np.random.uniform(low=-.5000000,high=.75000000,size=784)
    HL1Weights11=np.random.uniform(low=-.5000000,high=.75000000,size=784)
    HL1Weights12=np.random.uniform(low=-.5000000,high=.75000000,size=784)
    x1=np.dot(Neurons1,HL1Weights1)
    x2=np.dot(Neurons1,HL1Weights2)
    x3=np.dot(Neurons1,HL1Weights3)
    x4=np.dot(Neurons1,HL1Weights4)
    x5=np.dot(Neurons1,HL1Weights5)
    x6=np.dot(Neurons1,HL1Weights6)
    x7=np.dot(Neurons1,HL1Weights7)
    x8=np.dot(Neurons1,HL1Weights8)
    x9=np.dot(Neurons1,HL1Weights9)
    x10=np.dot(Neurons1,HL1Weights10)
    x11=np.dot(Neurons1,HL1Weights11)
    x12=np.dot(Neurons1,HL1Weights12)
    Sg1=Simgmoid(x1)
    Sg2=Simgmoid(x2)
    Sg3=Simgmoid(x3)
    Sg4=Simgmoid(x4)
    Sg5=Simgmoid(x5)
    Sg6=Simgmoid(x6)
    Sg7=Simgmoid(x7)
    Sg8=Simgmoid(x8)
    Sg9=Simgmoid(x9)
    Sg10=Simgmoid(x10)
    Sg11=Simgmoid(x11)
    Sg12=Simgmoid(x12)
    Neurons2.append(Sg1)
    Neurons2.append(Sg2)
    Neurons2.append(Sg3)
    Neurons2.append(Sg4)
    Neurons2.append(Sg5)
    Neurons2.append(Sg6)
    Neurons2.append(Sg7)
    Neurons2.append(Sg8)
    Neurons2.append(Sg9)
    Neurons2.append(Sg10)
    Neurons2.append(Sg11)
    Neurons2.append(Sg12)
    Neurons2=np.array(Neurons2)
    print(len(Neurons2))
    #print(Neurons2)
    #print(HL1Weights)
    HiddenLayer2(Neurons2)
def InputLayer(img):
    Neurons1=[]
    n=0
    nn=0
    while n<(img.shape[0]):
            Neuron=1-(img[n,nn]/np.amax(img))
            Neurons1.append(Neuron)
            if nn==(img.shape[1]-1):
                n+=1
                nn=0
                continue
            nn+=1
    Neurons1=np.array(Neurons1)
    print(len(Neurons1))
    #print(Neurons1)
    HiddenLayer1(Neurons1)
def ImageFormat(img):
    img.thumbnail((28,28))
    img=np.asarray(img,dtype='float')
    return img
def ImageGet():
    img = Image.open('1.jpg').convert('L')
    img=ImageFormat(img)
    InputLayer(img)
    return img
def Order():
    n=0
    while n<1000:
        img=ImageGet()
        n+=1
    """plt.imshow(img, cmap=plt.cm.binary,interpolation="nearest")
    plt.show()"""
Order()


"""def OutputLayer(Neurons3):
    global OWeights
    OBs=[]
    n=0
    Neurons4=[]
    while n!=10:
        OWeights=Weights2()
        x=np.dot(Neurons3,OWeights)
        Sg=Simgmoid(x)
        Neurons4.append(Sg)
        n+=1
    Neurons4=np.array(Neurons4)
    print(len(Neurons4))
    #print(Neurons4)
    Predictions(Neurons4)"""

"""def HiddenLayer1(Neurons1):
    global HL1Weights
    HL1Bs=0
    n=0
    Neurons2=[]
    while n!=12:
        HL1Weights=Weights1()
        x=np.dot(Neurons1,HL1Weights)
        Sg=Simgmoid(x)
        Neurons2.append(Sg)
        n+=1
    Neurons2=np.array(Neurons2)
    print(len(Neurons2))
    #print(Neurons2)
    #print(HL1Weights)
    HiddenLayer2(Neurons2)"""

"""def HiddenLayer2(Neurons2):
    global HL2Weights
    HL2Bs=0
    n=0
    Neurons3=[]
    while n!=12:
        HL2Weights=Weights1()
        x=np.dot(Neurons2,HL2Weights)
        Sg=Simgmoid(x)
        Neurons3.append(Sg)
        n+=1
    Neurons3=np.array(Neurons3)
    print(len(Neurons3))
    #print(Neurons3)
    #print(HL2Weights)
    HiddenLayer2(Neurons3)"""


"""def Weights2():
    Weights=np.random.uniform(low=-.5000000,high=.75000000,size=12)
    return Weights
def Weights1():
    Weights=np.random.uniform(low=-.5000000,high=.5000000,size=784)
    return Weights"""
