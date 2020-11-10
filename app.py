from flask import Flask, request, render_template
import matplotlib.pyplot as plt
from base64 import b64encode
import numpy as np
import cv2
import time
from PIL import Image
import io


app = Flask(__name__)


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=["POST","GET"])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    
    img = request.files['file']
    print(img)
    
    image_BGR = plt.imread(img)
    h,w = image_BGR.shape[:2]
    
    print(h,w)
    print(f'Image height {h} and width {w}')

    blob = cv2.dnn.blobFromImage(image_BGR, 1/255, (416,416), swapRB =True, crop= False)

    #check point
    print('Image shape: ',image_BGR.shape)
    print('Blob shape: ',blob.shape)

    #check point
    blob_to_show = blob[0,:,:,:].transpose(1,2,0)
    print('blob_to_show shape',blob_to_show.shape)

    
    with open('coco.names') as f:
        labels = [line.strip() for line in f]

    network = cv2.dnn.readNetFromDarknet('yolov3.cfg',
                                         'yolov3.weights')

    layers_names_all = network.getLayerNames()
 
    layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]
    
    
    
    probability_minimum = 0.5
    
    threshold = 0.3
    
    colours = np.random.randint(0,255, size = (len(labels) ,3), dtype = 'uint8')
    
    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()
    
    print(f'object detection took {end - start} seconds')
    
    bounding_boxes = []
    confidences = []
    class_numbers = []
    
    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            
            #check point
            #print(detected_objects.shape)
            
            if  confidence_current > probability_minimum:
                box_current = detected_objects[0:4] * np.array([w,h,w,h])
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width/2))
                y_min = int(y_center - (box_height/2))
                
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
                
        
        result = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
        
        counter = 1
        if len(result)>0:
            
            for i in result.flatten():
                print(f'object {counter}: {labels[int(class_numbers[i])]}')
                counter +=1
                
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                
                colour_box_current = colours[class_numbers[i]].tolist()
                
        
                
                cv2.rectangle(image_BGR, (x_min,y_min),
                              (x_min + box_width, y_min+box_height),
                              colour_box_current,2)
                
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                  confidences[i])
                
                cv2.putText(image_BGR, text_box_current, (x_min, y_min-5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)
                
                print()
                print('Total objects been detected: ',len(bounding_boxes))
                print('Number of objects left after non-maximum supprssion: ',counter-1)
                
    
    
    
    #im = Image.fromarray(image_BGR)
    #im.save('static/pics/temp1.jpg')
    file_object = io.BytesIO()
    img= Image.fromarray(image_BGR.astype('uint8'))
    img.save(file_object, 'PNG')
    base64img = "data:image/png;base64,"+b64encode(file_object.getvalue()).decode('ascii')
    
    #im.save('static/pics/temp1.jpg')
    return render_template('index.html', user_image=base64img)

if __name__ == "__main__":
    app.run(debug=True)
