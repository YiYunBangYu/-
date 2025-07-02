import numpy as np

def going_on(z):
    return z

def argmaximum(z):
    result = np.zeros(z.shape)
    result[np.argmax(z)] = 1
    return result

def threhold(z):
    for i in range(z.shape[1]):
        if z[i] > 0.7: z[i]=1
        else:z[i] = 0

class sigmoid:
    @staticmethod
    def f(z):
        return 1.0/(1.0+np.exp(-z))
    @staticmethod
    def d(z):
        m = 1.0/(1.0+np.exp(-z))
        return m(1-m)
    
Sig = sigmoid
    
class ReLu:
    @staticmethod
    def f(z):
        return np.maximum(0,z)
    @staticmethod
    def d(z):
        return np.where(z>0,1.0,0.0)

Re = ReLu

class MSE:
    @staticmethod
    def f(a,y):
        return ((a-y).T @ (a-y))/2
    @staticmethod
    def d(a,y):
        return a-y
    
class BCE:
    @staticmethod
    def f(a,y):
        return y.T @ np.log(a) + (1-y.T) @ np.log(1-a)
    @staticmethod
    def d(a,y):
        return (a-y)/(a*(1-a))              #I don't dispose the error log(0) for I think it's almost impossible.

class Layer:

    def __init__(self,size,inp_num,activation):
        self.W = np.random.randn(size,inp_num)
        self.b = np.zeros((size,1))
        self.activation = activation
        self.z = None
        self.a = None

class Network:

    def __init__(self,*args,cost_function = MSE,outputmethod = argmaximum):   #arg[even]:size of the layer;arg[odd]:activation of the layer.
        self.layers = []
        input_layer = Layer(args[0],1,args[1]) 
        self.layers.append(input_layer)

        for i in range(2 , len(args)-2 , 2):
            new_layer = Layer(args[i],args[i-2],args[i+1])
            self.layers.append(new_layer)
        self.layers.append(Layer(args[-2],args[-4],args[-1]))

        self.output = np.zeros((args[-2],1))
        self.output_method = outputmethod
        self.num_layers = len(self.layers)
        self.C = cost_function

    def feedforward(self,X,output=True):
        output = X
        for i in range(self.num_layers):
            z = np.dot(self.layers[i].W , output) + self.layers[i].b
            output = self.layers[i].activation.f(z)
            self.layers[i].z = z
            self.layers[i].a = output
        if output:
            self.output = self.output_method(self.layers[-1].a)
    
    def SGD(self,X_set,Y_set,epoch,mini_batch_size,alpha):      #X-set is a matrix composed of column vectors of training data.
        set_size = X_set.shape[1]
        
        for i in range(epoch):
            permutation = np.random.permutation(set_size)
            X_n = X_set[:,permutation]
            Y_n = Y_set[:,permutation]
            mini_batches = [X_n[:,k:k+mini_batch_size] for k in range(0,set_size,mini_batch_size)]
            Y_batches = [Y_n[:,k:k+mini_batch_size] for k in range(0,set_size,mini_batch_size)]
            
            for mini_batch ,Y_batch in zip(mini_batches,Y_batches):                #backpropogation
                self.feedforward(mini_batch,False)         #The last row as the expected output.
                
                delta = self.C.d(self.layers[-1].a,Y_batch) *self.layers[-1].activation.d(self.layers[-1].z)
                dW = (delta @ self.layers[-2].a.T) / mini_batch_size
                db = np.mean(delta,axis=0,keepdims=True)
                self.layers[-1].W -= alpha*dW
                self.layers[-1].b -= alpha*db

                for l in range(2 , self.num_layers):
                    delta = (self.layers[-l+1].W.T @ delta)*self.layers[-l].activation.d(self.layers[-l].z)
                    dW = (delta @ self.layers[-l-1].a.T) / mini_batch_size
                    db = np.mean(delta,axis=0,keepdims=True)
                    self.layers[-l].W -= alpha*dW
                    self.layers[-l].b -= alpha*db

                delta = (self.layers[1].W.T @ delta)*self.layers[0].activation.d(self.layers[0].z)
                dW = (delta @ self.layers[-l-1].a.T) / mini_batch_size
                db = np.mean(delta,axis=0,keepdims=True)
                self.layers[0].W -= alpha*dW
                self.layers[0].b -= alpha*db
            
            print(f"Epoch:{i}")

    def evaluate(self,X_T,Y_T):
        volume = X_T.shape[1]

        num = 0
        self.feedforward(X_T)
        for i in range(Y_T.shape[0]):
            if(self.output[i]==Y_T[i]):num += 1
        print(f" Succeed:{num} / {volume}")

