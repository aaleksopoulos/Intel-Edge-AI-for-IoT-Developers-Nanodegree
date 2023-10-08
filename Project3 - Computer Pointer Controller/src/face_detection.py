from src.model import Model, DEBUG

class Model_Face_Detection(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_path, device='CPU', extensions=None, prob_threshold=0.6):
        '''
        TODO: Use this to set your instance variables.
        '''
        #raise NotImplementedError
        Model.__init__(self, model_path=model_path, device=device, extensions=extensions, prob_threshold=prob_threshold)
        self.model_name = 'Face Detection'
        self.model_path = model_path
        self.model_structure = model_path+'.xml'
        self.model_weights = model_path+'.bin'

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        #raise NotImplementedError
        prep_img = self.preprocess_input(image)

        output_frame = self.exec_net.infer({self.input_blob : prep_img})

        tracked_list = self.preprocess_output(output_frame, image)
        return tracked_list


    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        #raise NotImplementedError
        #get the image width and height
        height = image.shape[0]
        width = image.shape[1]
        
        tracked_list = [] #to keep track what the model tracked

        for fr in outputs[self.output_blob][0][0]:

            if (fr[0] == -1):
                break

            if (fr[2]>=self.prob_threshold): #if the probability is above the one stated
                x1 = int(fr[3]*width)
                y1 = int(fr[4]*height)
                x2 = int(fr[5]*width)
                y2 = int(fr[6]*height)
                if DEBUG:
                    print("--------------------------")
                    print("calucalated x1: ", x1)
                    print("calucalated x2: ", x2)
                    print("calucalated y1: ", y1)
                    print("calucalated y2: ", y2)
                    print("--------------------------")
                tracked_list.append([x1, y1, x2, y2])

        if len(tracked_list)==0:
            return None

        return tracked_list
