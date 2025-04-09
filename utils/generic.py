import numpy as np
import cv2

class DepthApproximation:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def depth(self, predictions, depth):
        for pred in predictions:
            check = np.inf
            for x in range(pred['x1'],pred['x2']):
                for y in range(pred['y1'],pred['y2']):
                    if depth[y][x] < check:
                        check = depth[y][x]
            pred.update({'estimated_depth': self.decode_depth(check)})
        return predictions

    def annotate_depth_on_img(self, img, predictions):
        for pred in predictions:
            cv2.rectangle(img=img, pt1=(int(pred['x1']), int(pred['y1'])),
                              pt2=(int(pred['x2']), int(pred['y2'])),
                              color=(255, 0,255), thickness=1)
            cv2.putText(img, str(round(pred['estimated_depth'], 3)), (int(pred['x1']), int(pred['y1'])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 1, cv2.LINE_AA, False)
        return img

    def decode_depth(self, val):
        NewValue = val*self.max_depth/255
        return NewValue        
