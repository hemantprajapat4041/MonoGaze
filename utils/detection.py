from ultralytics import YOLO

class Yolo2DObjectDetection:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self,image):
        model = self.model
        results = model.predict(image)
        names = model.names
        predictions = []
        for result in results[0].boxes.xyxy:
            predictions.append({
                'x1': int(result[0].item()),
                'y1': int(result[1].item()),
                'x2': int(result[2].item()),
                'y2': int(result[3].item()),
            })

        for i in range(len(predictions)):
            predictions[i]['class'] = names[int((results[0].boxes.cls[i]))]
        return predictions
    
    def lanczos_conversion(self, predictions, original_size, conversion_size):
        orig_width, orig_height = original_size
        depth_width, depth_height = conversion_size
        
        # Calculate scaling factors
        width_scale = depth_width / orig_width
        height_scale = depth_height / orig_height
        for pred in predictions:
        
        # Unpack the original coordinates
            x1, y1, x2, y2 = pred['x1'], pred['y1'], pred['x2'], pred['y2']
            
            # Scale the coordinates
            x1_scaled = int(x1 * width_scale)
            y1_scaled = int(y1 * height_scale)
            x2_scaled = int(x2 * width_scale)
            y2_scaled = int(y2 * height_scale)
            
            # Ensure coordinates stay within bounds
            pred['x1'] = max(0, min(x1_scaled, depth_width - 1))
            pred['y1'] = max(0, min(y1_scaled, depth_height - 1))
            pred['x2'] = max(0, min(x2_scaled, depth_width - 1))
            pred['y2'] = max(0, min(y2_scaled, depth_height - 1))
        
        return predictions