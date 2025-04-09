from ultralytics import YOLO

class Yolo2DObjectDetection:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self,image):
        model = self.model
        results = model.predict(image)
        predictions = []
        for result in results[0].boxes.xyxy:
            predictions.append({
                'x1': int(result[0].item()),
                'y1': int(result[1].item()),
                'x2': int(result[2].item()),
                'y2': int(result[3].item())
            })
        return predictions