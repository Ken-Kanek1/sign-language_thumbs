import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Create a function to check thumb positions
def check_thumb_position(landmarks):
    # Extract thumb landmarks
    thumb_landmarks = landmarks[0:4]

    # Calculate the difference in Y-coordinates between the tip and base of the thumb
    thumb_y_diff = thumb_landmarks[3].y - thumb_landmarks[0].y

    # Define a threshold for thumb position
    thumb_threshold = 0.02  # Adjust as needed

    # Check if the thumb is up or down based on its position
    if thumb_y_diff < -thumb_threshold:
        return "Like"
    elif thumb_y_diff > thumb_threshold:
        return "Dislike"
    else:
        return "Neutral"

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Check thumb position and display feedback
            thumb_position = check_thumb_position(landmarks.landmark)
            cv2.putText(frame, thumb_position, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Thumbs Feedback', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
