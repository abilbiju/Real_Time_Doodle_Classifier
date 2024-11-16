import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import pyttsx3

class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}

# List of class names
class_names = ["",
    "Apple", "Butterfly", "Bus", "Circle", "Cloud",
    "Cup", "Eye", "Fish", "Flower", "Headphone",
    "House", "Ice cream", "Ladder", "Leaf", "Lolipop",
    "Mountain", "Necklace", "Pencil", "Smiley face", "Square",
    "Star", "Sun", "Table", "Takshak logo", "Television",
    "Tree", "Triangle", "Umbrella", "Vase", "Wine glass"
]

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the trained model (.h5 file)
model = tf.keras.models.load_model("keras_Model.h5", compile=False, custom_objects=custom_objects)

# Giving different arrays to handle colour points of different colour
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# The kernel to be used for dilation purpose
kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Set up Canvas
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

def classify_drawing(image):
    """
    Function to classify the drawing using the trained model and return the class name.
    """
    global model, class_names
    # Resize the image to match the model input
    image = cv2.resize(image, (224, 224))  # Change (224, 224) to your model's input size
    
    # Ensure the image has 3 channels
    if len(image.shape) == 2:
        # If the image is grayscale, convert to 3 channels
              image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # If the image has an alpha channel, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Image is already RGB
        pass
    else:
        raise ValueError("Unexpected image format")
    
    image = image.astype(np.float32) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict the class          b                        b    of the image
    predictions = model.predict(image)
    class_index = predictions.argmax()  # Get the class with the highest probability
    
    # Get the class name
    class_name = class_names[class_index]
    return class_name

def speak_classification(class_name):
    """
    Function to speak the classification result.
    """
    engine.say(f"The drawing is classified as {class_name}")
    engine.runAndWait()

ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    # Convert to RGB
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw the color selection rectangles on the frame
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark predictiocn
    result = hands.process(framergb)

    # If landmarks are detected
    if result.multi_hand_landmarks:
        landmarks = []
        for handLms in result.multi_hand_landmarks:
            for lm in handLms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            # Draw the landmarks
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

        if len(landmarks) > 8:
            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])

            # Check for drawing gesture
            if (thumb[1] - center[1] < 30):
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1

            # Check for selecting color or clear action
            elif center[1] <= 65:
                if 40 <= center[0] <= 140:  # Clear Button
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]
                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0
                    paintWindow[67:, :, :] = 255
                elif 160 <= center[0] <= 255:
                    colorIndex = 0  # Blue
                elif 275 <= center[0] <= 370:
                    colorIndex = 1  # Green
                elif 390 <= center[0] <= 485:
                    colorIndex = 2  # Red
                elif 505 <= center[0] <= 600:
                    colorIndex = 3  # Yellow

            else:
                # Add points to the corresponding color deque
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)

    # Draw the lines on the frame
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Display the frames
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    # Press 'q' to quit and 'c' to classify the drawing
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        try:
            paintWindow = paintWindow.astype(np.uint8)
            class_name = classify_drawing(paintWindow)
            print(f"Classified as: {class_name}")
            speak_classification(class_name)
        except Exception as e:
            print(f"Error during classification: {e}")

# Release resources
cap.release()
cv2.destroyAllWindows()
