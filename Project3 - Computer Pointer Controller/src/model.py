import os, cv2
from openvino.inference_engine import IECore

DEBUG = False #helper attribute

class Model:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_path, device='CPU', extensions=None, prob_threshold=0.6):
        '''
        TODO: Use this to set your instance variables.
        '''
        #raise NotImplementedError
        self.model_name = "Parent_Class"
        self.model_path= None #we cannot use this class
        self.device = device
        self.extensions = extensions
        self.model_structure = model_path+'.xml'
        self.model_weights = model_path+'.bin'
        self.prob_threshold = prob_threshold
        #check if we have provided a valid file for the model files
        if not self.check_model():
            exit(1)

        #initialize the Inference Engine and get the instance of executable network
        self.core = IECore()
        self.network = self.core.read_network(model=self.model_structure, weights=self.model_weights)

        #get the input and output blobs
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        #get the shape of the input and output
        self.input_shape = self.network.inputs[self.input_blob].shape
        self.output_shape = self.network.outputs[self.output_blob].shape


    def get_unsupported_layers(self):
        '''
        Returns a list of the unsupported layers
        NOTE For OpenVINO version 2020 and above, the cpu_extension is not needed
        '''
        #get a list of the supported layers
        supported_layers = self.core.query_network(self.network, device_name=self.device)
        #get the required layers
        required_layers = list(self.network.layers.keys())
        #check if there are unsupported layers
        unsupported_layers = []
        for layer in required_layers:
            if layer not in supported_layers:
                unsupported_layers.append(layer)

        return unsupported_layers

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        #raise NotImplementedError


        self.exec_net = self.core.load_network(network=self.network, device_name=self.device, num_requests=1)

        #check if there are any unsupported layers
        unsupported_layers = self.get_unsupported_layers()

        #if there are any unsupported layers, add CPU extension, if avaiable
        if (len(unsupported_layers)>0) and (self.device=='CPU'):
            print("There are unsupported layers found, will try to add CPU extension...")
            self.core.add_extension(extension_path=self.extensions, device=self.device)

        #add, if provided, a cpu extension
        if (self.extensions):
            self.core.add_extension(self.extensions)

        #recheck for unsupported layers, and exit if there are any
        unsupported_layers = self.get_unsupported_layers()
        if (len(unsupported_layers)>0) and (self.device!='CPU'):
            print("There are  unsupported layers, exiting...")
            exit(1)
        if (len(unsupported_layers)>0):
            print("After adding CPU extension, there are still unsupported layers, exiting...")
            exit(1)
        
        #load to network to get the executable network
        self.exec_network = self.core.load_network(self.network, self.device)

        return self.exec_network

    def check_model(self):
        '''
        If the path to the model xml and bin files exists, returns True, else False
        '''
        #raise NotImplementedError
        if ((os.path.exists(self.model_structure)) and (os.path.exists(self.model_weights))):
            if DEBUG:
                print("model found")
                print("model_xml: ", self.model_structure)
                print("model_bin: ", self.model_weights)
                print("device: ", self.device)
            return True
        else:
            print("There was a problem reading the xml file provided, exiting...")
            return False

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''     
        #raise NotImplementedError
        img = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        img = img.transpose((2,0,1))
        return img.reshape(1, *img.shape)
