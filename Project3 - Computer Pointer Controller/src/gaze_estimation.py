from src.model import Model, DEBUG

class Model_Gaze_Estimation(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_path, device='CPU', extensions=None, prob_threshold=0.6):
        '''
        TODO: Use this to set your instance variables.
        '''
        #raise NotImplementedError
        Model.__init__(self, model_path=model_path, device=device, extensions=extensions, prob_threshold=prob_threshold)
        self.model_name = 'Gaze_Estimation'
        self.model_path = model_path
        self.model_structure = model_path+'.xml'
        self.model_weights = model_path+'.bin'
        self.input_blob = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_blob[1]].shape
        self.output_blob = [i for i in self.network.outputs.keys()]

    def predict(self, left_eye, right_eye, head_pose_angle):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        #raise NotImplementedError
        prep_left_eye = self.preprocess_input(left_eye)
        pre_right_eye = self.preprocess_input(right_eye)

        output_frame = self.exec_net.infer({'left_eye_image':prep_left_eye, 'right_eye_image':pre_right_eye, 'head_pose_angles':head_pose_angle})

        gaze_vector = self.preprocess_output(output_frame)
        return gaze_vector

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        #raise NotImplementedError

        gaze_vector = outputs[self.output_blob[0]][0]

        return gaze_vector