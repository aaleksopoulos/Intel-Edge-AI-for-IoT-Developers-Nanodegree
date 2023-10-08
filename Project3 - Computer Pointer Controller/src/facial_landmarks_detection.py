from src.model import Model, DEBUG

class Model_Facial_Landmarks_Detection(Model):
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
        left_eye, right_eye, eye_box_coords = self.preprocess_output(output_frame, image)
        
        return left_eye, right_eye, eye_box_coords 


    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        #raise NotImplementedError
        #get the image width and height
        height = image.shape[0]
        width = image.shape[1]

        offset = 15

        left_eye = []
        right_eye = []
        eye_box_coords = [] 


        #get the center of each eye
        left_eye_x = int(outputs[self.output_blob][0][0][0][0]*width)
        left_eye_y = int(outputs[self.output_blob][0][1][0][0]*height)

        right_eye_x = int(outputs[self.output_blob][0][2][0][0]*width)
        right_eye_y = int(outputs[self.output_blob][0][3][0][0]*height)

        #get the coordinates of each bounding box
        right_eye_x1 = right_eye_x-offset
        right_eye_x2 = right_eye_x+offset
        right_eye_y1 = right_eye_y-offset
        right_eye_y2 = right_eye_y+offset

        left_eye_x1 = left_eye_x-offset
        left_eye_x2 = left_eye_x+offset
        left_eye_y1 = left_eye_y-offset
        left_eye_y2 = left_eye_y+offset

        left_eye = image[left_eye_y1:left_eye_y2, left_eye_x1:left_eye_x2]
        right_eye = image[right_eye_y1:right_eye_y2, right_eye_x1:right_eye_x2]
        eye_box_coords = [[left_eye_x1, left_eye_y1,left_eye_x2, left_eye_y2], [right_eye_x1, right_eye_y1, right_eye_x2, right_eye_y2]]

        if DEBUG:
            print("--------------------------")
            print("calucalated left_eye_x: ", left_eye_x)
            print("calucalated left_eye_y: ", left_eye_y)
            print("calucalated right_eye_x: ", right_eye_x)
            print("calucalated right_eye_y: ", right_eye_y)
            print("--------------------------")
   
        return left_eye, right_eye, eye_box_coords
