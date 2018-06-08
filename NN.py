import numpy as np


class ActivationFunction:
    def __init__(self,func, dfunc) :
        self.func = func
        self.dfunc = dfunc
  

sigmoid = lambda x: 1 / (1 + np.exp(-x))

sigmoidPrime = lambda z: np.multiply(np.array(sigmoid(np.array(z))), np.array((1 - sigmoid(np.array(z)))))

sigmoidAF = ActivationFunction(sigmoid,sigmoidPrime)

tanh =ActivationFunction(np.tanh,lambda x: 4 / (np.power(np.exp(-x) + np.exp(x), 2)))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class NeuralNetwork :
    def __init__(self,a, b, c=10) :
        if isinstance(a,NeuralNetwork) :
           self.input_layer = a.input_layer
           self.hidden_layer = a.hidden_layer 
           self.output_layer = a.output_layer

           self.weights_ih = a.weights_ih
           self.weights_ho = a.weights_ho

           self.bias_h = a.bias_h
           self.bias_o = a.bias_o
        else :
           self.input_layer = a
           self.hidden_layer = b 
           self.output_nodes = c
           glorot_init = np.sqrt(6 / (1.0 * (self.hidden_layer + self.input_layer)))
           self.weights_ih =np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (self.hidden_layer, self.input_layer)))
           self.weights_ho = np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (self.output_layer,self.hidden_layer)))
           self.weights_ih.randomize() 
           self.weights_ho.randomize() 

           self.bias_h =np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, self.hidden_layer)))
           self.bias_o = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, self.output_layer)))
           self.bias_h.randomize() 
           self.bias_o.randomize() 
        self.lr=self.setLearningRate()
        self.activation_function=ActivationFunction(sigmoid,sigmoidPrime)

    def setLearningRate(learning_rate=0.1):
        lr = learning_rate
    
    def setActivationFunction(func = sigmoid) :
        activation_function = func 

def predict(input_array) :

     # Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array) 
    let hidden = Matrix.multiply( self.weights_ih, inputs) 
    hidden.add( self.bias_h) 
     # activation function!
    hidden.map( self.activation_function.func) 

     # Generating the output's output!
    let output = Matrix.multiply( self.weights_ho, hidden) 
    output.add( self.bias_o) 
    output.map( self.activation_function.func) 

     # Sending back to the caller!
    return output.toArray() 
   

  

  
   

def train(input_array, target_array) :
     # Generating the Hidden Outputs
    inputs = Matrix.fromArray(input_array) 
    hidden = Matrix.multiply( self.weights_ih, inputs) 
    hidden.add( self.bias_h) 
     # activation function!
    hidden.map( self.activation_function.func) 

     # Generating the output's output!
    let outputs = Matrix.multiply( self.weights_ho, hidden) 
    outputs.add( self.bias_o) 
    outputs.map( self.activation_function.func) 

     # Convert array to matrix object
    let targets = Matrix.fromArray(target_array) 

     # Calculate the error
     # ERROR = TARGETS - OUTPUTS
    let output_errors = Matrix.subtract(targets, outputs) 

     # let gradient = outputs * (1 - outputs) 
     # Calculate gradient
    let gradients = Matrix.map(outputs,  self.activation_function.dfunc) 
    gradients.multiply(output_errors) 
    gradients.multiply( self.learning_rate) 


     # Calculate deltas
    let hidden_T = Matrix.transpose(hidden) 
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T) 

     # Adjust the weights by deltas
     self.weights_ho.add(weight_ho_deltas) 
     # Adjust the bias by its deltas (which is just the gradients)
     self.bias_o.add(gradients) 

     # Calculate the hidden layer errors
    let who_t = Matrix.transpose( self.weights_ho) 
    let hidden_errors = Matrix.multiply(who_t, output_errors) 

     # Calculate hidden gradient
    let hidden_gradient = Matrix.map(hidden,  self.activation_function.dfunc) 
    hidden_gradient.multiply(hidden_errors) 
    hidden_gradient.multiply( self.learning_rate) 

     # Calcuate input->hidden deltas
    let inputs_T = Matrix.transpose(inputs) 
    let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T) 

     self.weights_ih.add(weight_ih_deltas) 
     # Adjust the bias by its deltas (which is just the gradients)
     self.bias_h.add(hidden_gradient) 

     # outputs.print() 
     # targets.print() 
     # error.print() 
   

  serialize() :
    return JSON.stringify(this) 
   

  static deserialize(data) :
    if (typeof data == 'string') :
      data = JSON.parse(data) 
     
    let nn = new NeuralNetwork(data.input_layer, data.hidden_layer, data.output_nodes) 
    nn.weights_ih = Matrix.deserialize(data.weights_ih) 
    nn.weights_ho = Matrix.deserialize(data.weights_ho) 
    nn.bias_h = Matrix.deserialize(data.bias_h) 
    nn.bias_o = Matrix.deserialize(data.bias_o) 
    nn.learning_rate = data.learning_rate 
    return nn 
   


   # Adding function for neuro-evolution
  copy() :
    return new NeuralNetwork(this) 
   

  mutate(rate) :
     # This is how we adjust weights ever so slightly
    function mutate(x) :
      if (Math.random() < rate) :
        var offset = randomGaussian() * 0.5 
         # var offset = random(-0.1, 0.1) 
        var newx = x + offset 
        return newx 
        else :
        return x 
       
     
     self.weights_ih.map(mutate) 
     self.weights_ho.map(mutate) 
     self.bias_h.map(mutate) 
     self.bias_o.map(mutate) 
   



 