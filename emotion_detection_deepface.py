# Import necessary libraries
import cv2
import numpy as np
import time
from deepface import DeepFace
from collections import Counter, deque
import matplotlib.pyplot as plt
import random

# Initialize emotion tracking counters
emotion_counter = Counter()  # Counts total occurrences of each emotion
emotion_history = deque(maxlen=10)  # Holds recent emotions for smoothing
emotion_durations = Counter()  # Tracks cumulative duration for each emotion
session_start_time = time.time()  # Session start timestamp
emotion_start_time = time.time()  # Start time of current emotion
current_emotion = None  # Tracks the most recent detected emotion
feedback_last_update = time.time()  # Timestamp for the last feedback update

# Define colors for overlay based on emotions
emotion_colors = {
    "happy": (0, 255, 0),       # Green for happy
    "sad": (255, 0, 0),         # Red for sad
    "angry": (0, 0, 255),       # Blue for angry
    "surprise": (255, 255, 0)   # Yellow for surprise
}

# AI feedback function based on the detected emotion
def ai_feedback(emotion):
    """Provides feedback based on the current emotion."""
    responses = {
        "happy": ["Keep up the positive energy!", "Happiness looks great on you!"],
        "sad": ["Take a deep breath. It’s okay to feel down sometimes.", "Consider doing something you enjoy!"],
        "angry": ["Take a few calming breaths.", "It might help to take a quick walk!"],
        "surprise": ["You look pleasantly surprised!", "Stay curious!"],
    }
    if emotion in responses:
        return random.choice(responses[emotion])
    return "Stay balanced and keep going!"

# Initialize video capture for real-time emotion detection
cap = cv2.VideoCapture(0)

# Main loop for emotion detection and feedback
while True:
    # Capture frame-by-frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video frame capture fails

    # Detect emotion in the frame using DeepFace
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    detected_emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
    emotion_history.append(detected_emotion)  # Add detected emotion to history

    # Determine the most frequent emotion in recent history for stability
    most_common_emotion = Counter(emotion_history).most_common(1)[0][0]

    # Update the main emotion and feedback only if the detected emotion changes steadily
    if most_common_emotion != current_emotion and time.time() - feedback_last_update > 5:  # 5-second cooldown
        # Calculate the duration of the previous emotion
        if current_emotion is not None:
            duration = time.time() - emotion_start_time
            emotion_durations[current_emotion] += duration

        # Update the current emotion and reset the timing
        current_emotion = most_common_emotion
        emotion_start_time = time.time()
        feedback_last_update = time.time()  # Reset feedback update cooldown

    # Update emotion counter
    emotion_counter[current_emotion] += 1

    # Apply color overlay based on the detected emotion
    if current_emotion in emotion_colors:
        overlay_color = emotion_colors[current_emotion]
        overlay = np.full(frame.shape, overlay_color, dtype=np.uint8)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Display the current emotion and feedback text on the video frame
    feedback_text = ai_feedback(current_emotion)
    cv2.putText(frame, f"Emotion: {current_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Feedback: {feedback_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the video frame with overlays and feedback
    cv2.imshow("Steady Emotion Detection", frame)

    # Press 'q' to quit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# End-of-session processing
# Calculate the duration for the last emotion displayed
if current_emotion is not None:
    emotion_durations[current_emotion] += time.time() - emotion_start_time

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()

# Generate and display session summary report
print("\n--- Session Summary ---")
total_duration = time.time() - session_start_time
for emotion, duration in emotion_durations.items():
    print(f"{emotion.capitalize()} Duration: {duration:.2f} seconds")

# Visualize Emotion Timeline with a Bar Chart
plt.figure(figsize=(10, 6))
plt.title("Emotion Timeline")
plt.xlabel("Emotion")
plt.ylabel("Frequency")
plt.bar(emotion_counter.keys(), emotion_counter.values(), color='skyblue')
plt.show()

print("\nThank you for using the Emotion Detection System. Have a great day!")
