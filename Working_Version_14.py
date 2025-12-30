import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp

UNITS = 256  # Transformer

# Initializers
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
INIT_ZEROS = tf.keras.initializers.constant(0.0)

# Activations
GELU = tf.keras.activations.gelu

def scaled_dot_product(q, k, v, softmax):
    # Calculates Q . K(transpose)
    qkt = tf.matmul(q, k, transpose_b=True)
    # Calculates scaling factor
    dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=tf.float32))
    scaled_qkt = qkt / dk
    softmax_output = softmax(scaled_qkt)

    z = tf.matmul(softmax_output, v)
    # Shape: (m, Tx, depth), same shape as q, k, v
    return z

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_of_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads
        self.wq = [tf.keras.layers.Dense(self.depth) for _ in range(num_of_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth) for _ in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth) for _ in range(num_of_heads)]
        self.wo = tf.keras.layers.Dense(d_model)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(scaled_dot_product(Q, K, V, self.softmax))
        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention


@tf.keras.saving.register_keras_serializable()
class Augmentation(tf.keras.layers.Layer):
    def __init__(self, noise_std):
        super(Augmentation, self).__init__()
        self.noise_std = noise_std

    def add_noise(self, t):
        B = tf.shape(t)[0]  # Batch size
        return tf.where(
            t == 0.0,  # If value is zero (pad), don't add noise
            0.0,
            t + tf.random.normal([B, 1, 1, tf.shape(t)[3]], 0, self.noise_std),  # Add Gaussian noise
        )

    def call(self, lips0, left_hand0, pose0, right_hand0, training=False):
        if training:
            lips0 = self.add_noise(lips0)  # Augment lips
            left_hand0 = self.add_noise(left_hand0)  # Augment left hand
            pose0 = self.add_noise(pose0)  # Augment pose
            right_hand0 = self.add_noise(right_hand0)  # Augment right hand
        return lips0, left_hand0, pose0, right_hand0


@tf.keras.saving.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self, num_blocks):
        super(Transformer, self).__init__(name='transformer')
        self.num_blocks = num_blocks
        self.mhas = []
        self.mlps = []

        for i in range(self.num_blocks):
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(UNITS, 8))  # Modify the arguments as needed
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([ 
                tf.keras.layers.Dense(UNITS, activation=GELU, kernel_initializer=INIT_GLOROT_UNIFORM),
                tf.keras.layers.Dropout(0.30),
                tf.keras.layers.Dense(UNITS, kernel_initializer=INIT_HE_UNIFORM),
            ]))

    def call(self, x):
        for mha, mlp in zip(self.mhas, self.mlps):
            x = x + mha(x)
            x = x + mlp(x)

        return x


class SignLanguageTranslator:
    def __init__(self, model_path, labels_path, threshold=0.5, num_classes=100):
        # Load the trained model from the .keras file
        self.model = tf.keras.models.load_model(model_path)

        # Load the feature labels and define actions
        self.labels = np.load(labels_path)
        self.actions = np.unique(self.labels)  # Get unique labels to form the actions list
        self.threshold = threshold
        self.num_classes = num_classes

        # Initialize Mediapipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.sequence = []  # Store keypoints in a sequence
        self.sentence = []  # Store the sentence formed by signs
        self.predictions = []  # Store the predicted actions
        
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)

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

    def prob_viz(self, res, image, colors):
        top_indices = np.argsort(res)[-3:][::-1]  # Top 3 predictions in descending order
        for i, idx in enumerate(top_indices):
            action_text = f"{self.actions[idx]}: {res[idx]:.2f}"
            bar_x = 10
            bar_y = (i * 30) + 10
            text_y = bar_y + 20

            bar_width = int(res[idx] * 200)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), colors[i], -1)
            cv2.putText(image, action_text, (bar_x, text_y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return image

    def start(self):
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Make detections
                image, results = self.mediapipe_detection(frame, holistic)

                # Extract keypoints and add to sequence
                keypoints = self.extract_keypoints(results)
                keypoints = keypoints[:104]  # Truncate if too large
                keypoints = np.pad(keypoints, (0, 104 - len(keypoints)))  # Pad if too small

                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]  # Keep the last 30 frames for prediction

                # Process sequence when there are exactly 30 frames
                if len(self.sequence) == 30:
                    sequence_array = np.array(self.sequence)  # Convert sequence to NumPy array

                    if sequence_array.size != 30 * 104 * 3:
                        sequence_array = sequence_array.flatten()
                        if sequence_array.size < 30 * 104 * 3:
                            sequence_array = np.pad(sequence_array, (0, (30 * 104 * 3) - sequence_array.size))
                        else:
                            sequence_array = sequence_array[:30 * 104 * 3]

                    # Reshape to (30, 104, 3)
                    sequence_array = sequence_array.reshape((30, 104, 3))

                    # Prediction logic
                    if sequence_array.shape == (30, 104, 3):
                        res = self.model.predict(np.expand_dims(sequence_array, axis=0))[0]
                        self.predictions.append(np.argmax(res))

                        # Visualization logic
                        if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                            self.sentence.append(self.actions[np.argmax(res)])

                        self.sentence = self.sentence[-5:]  # Keep the last 5 actions

                        # Update image with predictions
                        image = self.prob_viz(res, image, colors=[(0, 255, 0), (0, 255, 255), (255, 0, 0)])

                # Show sentence
                cv2.putText(image, " ".join(self.sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Sign Language Translator", image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.cap.release()
            cv2.destroyAllWindows()

# Example of usage
model_path = 'E:/University/Year 5/Semester 2/Final Design/Phase 3/Sign Language to Text Translator Program/model_new_100.keras'
labels_path = 'E:/University/Year 5/Semester 2/Final Design/Phase 3/Sign Language to Text Translator Program/feature_labels.npy'

translator = SignLanguageTranslator(model_path, labels_path)
translator.start()
