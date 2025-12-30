import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('action_new_5.h5')

actions = np.load('E:/University/Year 5/Semester 2/Final Design/Phase 3/Sign Language to Text Translator Program/actions.npy', allow_pickle=True)

threshold = 0.5

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# Define shades of blue for confidence visualization
confidence_colors = [(255, 100, 100), (200, 150, 255), (150, 200, 255)]

def prob_viz(res, actions, image):
    top_indices = np.argsort(res)[-3:][::-1]  # Top 3 predictions in descending order
    for i, idx in enumerate(top_indices):
        action_text = f"{actions[idx]}: {res[idx]:.2f}"
        
        # Position text and bars in the top left corner of the screen
        bar_x = 10
        bar_y = (i * 30) + 10
        text_y = bar_y + 20

        # Draw the confidence bar in different shades of blue
        bar_width = int(res[idx] * 200)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), confidence_colors[i], -1)

        # Draw the text in smaller size above the bars
        cv2.putText(image, action_text, (bar_x, text_y - 5), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
    return image

def start_translation():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    sequence, sentence, predictions = [], [], []
    translation_started = False

    cv2.namedWindow('OpenCV Feed', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('OpenCV Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                sequence_array = np.array(sequence)
                if sequence_array.shape[1] == (468 * 3 + 21 * 3 + 21 * 3 + 33 * 4):  # Adjust based on model input shape
                    res = model.predict(np.expand_dims(sequence_array, axis=0))[0]
                    predictions.append(np.argmax(res))
                    
                    if np.unique(predictions[-10:])[0] == np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)]) 

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    # Visualize top 3 probabilities
                    image = prob_viz(res, actions, image)

            # Create a white box at the bottom for the sentence
            white_box_height = 40  # Height of the white box
            white_box_width = 600  # Width of the white box
            box_x = 10  # Starting X position
            box_y = image.shape[0] - white_box_height - 10  # Y position near the bottom

            # Draw the white box (smaller than the full width of the screen)
            cv2.rectangle(image, (box_x, box_y), (box_x + white_box_width, box_y + white_box_height), (255, 255, 255), -1)
            
            # Draw the sentence in the box with smaller text and black color
            cv2.putText(image, ' '.join(sentence), (box_x + 10, box_y + white_box_height // 2 + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            # Add a message indicating translation has started
            if not translation_started:
                cv2.putText(image, "Translation has started!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                translation_started = True

            # Show a message at the bottom that says "Press Escape to return"
            cv2.putText(image, "Press Escape to return", (10, image.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)
            
            # Check for 'Escape' key press to return to the menu
            if cv2.waitKey(10) & 0xFF == 27:  # 27 is the ASCII code for the Escape key
                break

    cap.release()
    cv2.destroyAllWindows()
