
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys

DEBUG = False

class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        #initialize it here to use the 2020 version
        self.ie = IECore()       
        try:
            #self.model=IENetwork(self.model_structure, self.model_weights)
            self.model = self.ie.read_network(model=self.model_structure, weights=self.model_weights )
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        
  
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: This method needs to be completed by you
        Use the initalized IECore instance to load the model
        '''
        self.net = self.ie.load_network(network=self.model, device_name=self.device, num_requests=1)
        #return self.net
        #raise NotImplementedError
        
    def predict(self, image):
        '''
        TODO: This method needs to be completed by you
        Returns a list of lists of the coordinates of the detected persons
        and the image with those rectangles
        '''
        prep_image = self.preprocess_input(image)
        
        input_dict={self.input_name:prep_image}
        
        infer_req = self.net.infer(inputs=input_dict)

        #the below is for async request
        #output = infer_req.outputs[self.output_name]
        output = infer_req[self.output_name]
        
        coords = self.preprocess_outputs(output, image)
        #get the output frame with the rectangles of the detected persons
        self.draw_outputs(coords, image)
        return coords, image
        #raise NotImplementedError

    
    def draw_outputs(self, coords, image):
        '''
        TODO: This method needs to be completed by you
        Draw rectangles for the detected persons, colored yellow
        and returns the image
        '''

        #loop over the ccords and draw the rectangle
        for coord in coords:
            x1 = coord[0]
            y1 = coord[1]
            x2 = coord[2]
            y2 = coord[3]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,255),1)
        #return image
        #raise NotImplementedError

    def preprocess_outputs(self, outputs, image):
        '''
        TODO: This method needs to be completed by you
        Loop over the outputs, and return the coordinates of any box above the threshold
        '''
        
        height = image.shape[0]
        width = image.shape[1]
        coords = [] #placeholder for the coords

        for out in outputs[0][0]:
            #print("len out[0][0]: ", len(out))
            #print('out thres:', out[2])
            if out[2] > self.threshold:
                
                x1 = int(out[3]*width)
                y1 = int(out[4]*height)
                x2 = int(out[5]*width)
                y2 = int(out[6]*height)
                coords.append([x1, y1, x2, y2])
                if DEBUG:
                    print('=======================================')
                    print('coords:' , coords)
        if DEBUG:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return coords
        #raise NotImplementedError

    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you
        Preprocess the image to fit with the model requirments
        and return the preprocessed image
        '''

        w = self.input_shape[2] 
        h = self.input_shape[3]
        img = cv2.resize(image, (h, w))
        img = img.transpose((2,0,1))
        return img.reshape(1, 3, w, h)
        #raise NotImplementedEror


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)