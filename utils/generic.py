import numpy as np
import cv2,os
import pandas as pd

class DepthApproximation:
    def __init__(self, max_depth=80):
        self.max_depth = max_depth

    def depth(self, predictions, depth, decode=True, eval=False, val=80):
        for pred in predictions:
            check = np.inf
            for x in range(pred['x1'],pred['x2']):
                for y in range(pred['y1'],pred['y2']):
                    if depth[y][x] < check:
                        check = depth[y][x]
            if decode and not eval:
                pred.update({'estimated_depth': self.decode_depth(check, val)})
            elif not decode and not eval:
                pred.update({'estimated_depth': check})
            if eval:
                pred.update({'actual_depth': check})

        return predictions

    def annotate_depth_on_img(self, img, predictions, eval=False):
        for pred in predictions:
            if pred['estimated_depth'] == np.inf:
                continue
            if eval:
                cv2.putText(img=img, text='AD ' + str(round(pred['actual_depth'], 3))+'m', org=(int(pred['x1']), int(pred['y2']-6)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1)
            else:
                cv2.rectangle(img=img, pt1=(int(pred['x1']), int(pred['y1'])),
                                pt2=(int(pred['x2']), int(pred['y2'])),
                                color=(255, 0,255), thickness=1)
                cv2.putText(img, 'PD ' + str(round(pred['estimated_depth'], 3))+ 'm', (int(pred['x1']), int(pred['y1']+12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA, False)
        return img

    def decode_depth(self, check, val):
        NewValue = check*val/255
        return NewValue        
    
    def evaluate(self, input_path, img, filename, predictions):
        name = os.path.splitext(os.path.basename(filename))[0]
        file_path = os.path.join(input_path, name + '.npy')
        depth_frame = np.load(file_path)
        actual_pred = self.depth(predictions, depth_frame, eval=True)
        img = self.annotate_depth_on_img(img, actual_pred, eval=True)
        return img, actual_pred

        
