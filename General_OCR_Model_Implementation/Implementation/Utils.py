
#output dims -> (1,x,x,1,5)

# boxes = decode_to_boxes(output)  output to boxes
# corner_boxes = boxes_to_corners(boxes) boxes to corners
# final_out = non_max_suppress(corner_boxes) 
#                   iou()



import numpy as np


def decode_to_boxes(output , ht , wd):
    #output : (x,x,1,5)
    #x,y,h,w
    img_ht = ht
    img_wd = wd
    threshold = 0.5
    grid_h,grid_w = output.shape[:2]
    final_boxes = []
    scores = []

    for i in range(grid_h):
        for j in range(grid_w):
            if output[i,j,0,0] > threshold:

                temp = output[i,j,0,1:5]
                
                x_unit = ((j + (temp[0]))/grid_w)*img_wd
                y_unit = ((i + (temp[1]))/grid_h)*img_ht
                width = temp[2]*img_wd*1.3
                height = temp[3]*img_ht*1.3
                
                final_boxes.append([x_unit - width/2,y_unit - height/2 ,x_unit + width/2,y_unit + height/2])
                scores.append(output[i,j,0,0])
    
    return final_boxes,scores



#Finds the Intersection over Union for two bounding boxes.
def iou(box1,box2):

    x_inter1 = max(box1[0],box2[0])
    y_inter1 = max(box1[1] ,box2[1])

    x_inter2 = min(box1[2],box2[2])
    y_inter2 = min(box1[3],box2[3])

    width_inter = x_inter2 - x_inter1
    height_inter = y_inter2 - y_inter1   
    area_intersection = width_inter * height_inter
    
    area_box1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0])*(box2[3] - box2[1])
    area_union = area_box1 + area_box2 - area_intersection
        
    iou = area_intersection/area_union
    
    return iou



def non_max(boxes , scores , iou_num):

    scores_sort = scores.argsort().tolist()
    keep = []
    
    while(len(scores_sort)):
        
        index = scores_sort.pop()
        keep.append(index)
        
        if(len(scores_sort) == 0):
            break
    
        iou_res = []
    
        for i in scores_sort:
            iou_res.append(iou(boxes[index] , boxes[i]))
        
        iou_res = np.array(iou_res)
        filtered_indexes = set((iou_res > iou_num).nonzero()[0])

        scores_sort = [v for (i,v) in enumerate(scores_sort) if i not in filtered_indexes]
    
    final = []
    
    for i in keep:
        final.append(boxes[i])
    
    return final


def decode(output , ht , wd , iou):
    boxes , scores = decode_to_boxes(output ,ht ,wd)
    boxes = non_max(boxes,np.array(scores) , iou)
    return boxes
    

