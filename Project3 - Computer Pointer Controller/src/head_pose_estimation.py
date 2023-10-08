from src.model import Model, DEBUG

class Model_Head_Pose_Estimation(Model):
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
        head_pose_angle = self.preprocess_output(outputs=output_frame)
        
        return head_pose_angle


    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        #raise NotImplementedError
        head_pose_angle = []
        head_pose_angle.append(outputs['angle_y_fc'][0][0])
        head_pose_angle.append(outputs['angle_p_fc'][0][0])
        head_pose_angle.append(outputs['angle_r_fc'][0][0])

        return head_pose_angle
