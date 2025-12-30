import cv2
import tensorflow as tf
import mediapipe as mp
import pandas as pd
import numpy as np

class StartTranslation:
    def __init__(self, model_path, actions_csv, threshold=0.5, num_classes=100):
        # Load model and actions
        self.model = tf.keras.models.load_model('action_new_5')
        self.actions = pd.read_csv(actions_csv)['sign'].values.tolist()
        self.threshold = threshold
        self.num_classes = num_classes
        
        # Initialize Mediapipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

        # Colors for visualization
        self.confidence_colors = [(255, 100, 100), (200, 150, 255), (150, 200, 255)]

        # Sentence and predictions
        self.sentence = []
        self.predictions = []

    def mediapipe_detection(self, image, holistic):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def extract_keypoints(self, results):
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(40 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        chest = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[0:22]]).flatten() if results.pose_landmarks else np.zeros(22 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([face, lh, chest, rh])

    def prob_viz(self, res, actions, image, colors):
        top_indices = np.argsort(res)[-3:][::-1]
        for i, idx in enumerate(top_indices):
            action_text = f"{actions[idx]}: {res[idx]:.2f}"
            bar_x = 10
            bar_y = (i * 30) + 10
            text_y = bar_y + 20
            bar_width = int(res[idx] * 200)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), colors[i], -1)
            cv2.putText(image, action_text, (bar_x, text_y - 5), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        main_action = actions[np.argmax(res)]
        main_confidence = res[np.argmax(res)]
        cv2.putText(image, f"Action: {main_action} ({main_confidence:.2f})", 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return image

    def start(self):
        sequence = []
        cap = cv2.VideoCapture(0)
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image, results = self.mediapipe_detection(frame, holistic)
                keypoints = self.extract_keypoints(results)
                keypoints = np.pad(keypoints, (0, 104 - len(keypoints)))
                sequence.append(keypoints)
                sequence = sequence[-30:]
                if len(sequence) == 30:
                    res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
                    self.predictions.append(np.argmax(res))
                    if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > self.threshold:
                            if len(self.sentence) > 0:
                                if self.actions[np.argmax(res)] != self.sentence[-1]:
                                    self.sentence.append(self.actions[np.argmax(res)])
                            else:
                                self.sentence.append(self.actions[np.argmax(res)])
                        if len(self.sentence) > 5:
                            self.sentence = self.sentence[-5:]
                        image = self.prob_viz(res, self.actions, frame, self.confidence_colors)
                cv2.imshow('OpenCV Feed', frame)
                if cv2.waitKey(10) == 27:  # ESC key
                    break
        cap.release()
        cv2.destroyAllWindows()


# Example Usage
translator = StartTranslation(
    model_path='action_new_5.keras',
    actions_csv='actions.npy'
)

translator.start()
