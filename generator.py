class Generator(keras.Model):
    
    def __init__(self, random_noise_size = 100):
        super().__init__(name='generator')
        #layers
        self.input_layer = keras.layers.Dense(units = random_noise_size)
        self.dense_1 = keras.layers.Dense(units = 4)
        self.leaky_1 = keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_2 = keras.layers.Dense(units = 8)
        self.leaky_2 = keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_3 = keras.layers.Dense(units = 16)
        self.leaky_3 = keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_4 = keras.layers.Dense(units = 32)
        self.leaky_4 = keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_5 = keras.layers.Dense(units = 64)
        self.leaky_5 = keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_6 = keras.layers.Dense(units = 128)
        self.leaky_6 = keras.layers.LeakyReLU(alpha = 0.01)
        self.output_layer = keras.layers.Dense(units=49152, activation = "tanh")
        
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.input_layer(input_tensor)
        x = self.dense_1(x)
        x = self.leaky_1(x)
        x = self.dense_2(x)
        x = self.leaky_2(x)
        x = self.dense_3(x)
        x = self.leaky_3(x)
        x = self.leaky_4(x)
        x = self.leaky_4(x)
        x = self.leaky_5(x)
        x = self.leaky_5(x)
        x = self.leaky_6(x)
        x = self.leaky_6(x)
        return  self.output_layer(x)
    
    def generate_noise(self,batch_size, random_noise_size):