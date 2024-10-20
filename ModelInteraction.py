import numpy as np
import tensorflow as tf
import cv2

# Define the class mapping
class_mapping = {
    0: 'Ariel', 1: 'Cadbury', 2: 'Choco Pie', 3: 'Dark Fantasy', 4: 'Dettol',
    5: 'Dove', 6: 'Gold', 7: 'Head and Shoulder', 8: 'Hide and Seek', 9: 'Jim Jam',
    10: 'Lipton', 11: 'Lux', 12: 'Nivea Deo', 13: 'OREO', 14: 'Pears', 
    15: 'Ponds fwsh', 16: 'Red Label', 17: 'Santoor', 18: 'Surf Excel', 
    19: 'Taj', 20: 'TeaVeda', 21: 'Tide', 22: 'CleanAndClear', 
    23: 'Fortune', 24: 'Fortune Soya Health', 25: 'MamaEarth', 26: 'Nivea fwsh'
}

# Load TensorFlow model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Run inference
def run_inference(model, input_data):
    output_data = model.predict(input_data)
    return output_data

# Smoothing function for bounding box coordinates
def smooth_bbox(x, y, w, h, prev_x, prev_y, prev_w, prev_h, smoothing_factor=0.2):
    if prev_x is not None:
        x = int(smoothing_factor * x + (1 - smoothing_factor) * prev_x)
        y = int(smoothing_factor * y + (1 - smoothing_factor) * prev_y)
        w = int(smoothing_factor * w + (1 - smoothing_factor) * prev_w)
        h = int(smoothing_factor * h + (1 - smoothing_factor) * prev_h)
    return x, y, w, h

# Main function
if __name__ == "__main__":
    model_path = r"cnnModel.h5"  # Replace with your model path

    # Load the model
    model = load_model(model_path)

    # Start video capture
    cap = cv2.VideoCapture(0)  # Use 0 for the primary camera

    # Initialize background subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    # Set minimum and maximum contour area
    MIN_CONTOUR_AREA = 10000
    MAX_CONTOUR_AREA = 50000  # Adjust based on your use case

    # Variables to store previous bounding box coordinates
    prev_x, prev_y, prev_w, prev_h = None, None, None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Apply background subtraction to detect moving objects
        foreground_mask = background_subtractor.apply(frame)

        # Detect contours on the foreground mask
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Proceed only if there is at least one valid contour
        if contours:
            # Find the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the area of the largest contour
            contour_area = cv2.contourArea(largest_contour)

            # Ensure the contour area is within the desired range
            if MIN_CONTOUR_AREA <= contour_area <= MAX_CONTOUR_AREA:
                # Find bounding box for the valid contour
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Smooth the bounding box coordinates
                x, y, w, h = smooth_bbox(x, y, w, h, prev_x, prev_y, prev_w, prev_h)

                # Update previous bounding box coordinates
                prev_x, prev_y, prev_w, prev_h = x, y, w, h

                # Crop and preprocess the region of interest (ROI)
                roi = frame[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, (400, 400))  # Resize to model's input size
                roi_normalized = roi_resized / 255.0  # Normalize to [0, 1]
                input_data = np.expand_dims(roi_normalized, axis=0).astype(np.float32)  # Add batch dimension

                # Run inference
                output_data = run_inference(model, input_data)

                # Get predicted class and confidence
                predicted_class_index = np.argmax(output_data[0])
                confidence = output_data[0][predicted_class_index]

                # Set a confidence threshold
                CONFIDENCE_THRESHOLD = 0.75
                if confidence > CONFIDENCE_THRESHOLD:
                    predicted_class_name = class_mapping.get(predicted_class_index, "Unknown Class")

                    # Draw a bounding box around the detected object
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Display the predicted class name and confidence near the bounding box
                    cv2.putText(frame, f'Predicted: {predicted_class_name}', (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'Confidence: {confidence:.2f}', (x, y+h+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the frame with the prediction and bounding box
        cv2.imshow("Live Video Feed", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()
