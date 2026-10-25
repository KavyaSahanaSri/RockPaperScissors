import cv2
import random
import time
import numpy as np

# Try to import MediaPipe; if unavailable, fallback to OpenCV-only heuristic
try:
    import mediapipe as mp
    # Check if MediaPipe has the old solutions API
    if hasattr(mp, 'solutions'):
        USE_MEDIAPIPE = True
    else:
        USE_MEDIAPIPE = False
        print("⚠️ Warning: MediaPipe version incompatible — using OpenCV fallback for gestures.")
except Exception:
    mp = None
    USE_MEDIAPIPE = False

# Initialize MediaPipe Hand module if available
if USE_MEDIAPIPE:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
else:
    print("⚠️ Warning: MediaPipe not available — using OpenCV fallback for gestures.")

# Game Variables
choices = ["Rock", "Paper", "Scissors"]
player_score = 0
computer_score = 0
gesture = "None"
timer = 0
state = "WAIT"
round_result = ""      # NEW: store the result of current round
result_display_time = 0  # NEW: time when result started displaying

# Instructions
INSTRUCTIONS = [
    "How to play:",
    "1) Press SPACE to start a 3-second countdown.",
    "2) Show Rock, Paper or Scissors to the camera.",
    "3) After countdown, computer picks and result displays.",
    "4) Press Q to quit and see final scores."
]

def choose_text_colors(frame, x, y, w, h):
    """Auto choose text colors based on background brightness."""
    h_frame, w_frame = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_frame, x + w), min(h_frame, y + h)
    if x0 >= x1 or y0 >= y1:
        return (255, 255, 255), (0, 0, 0)

    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean = int(gray.mean())
    if mean < 120:
        return (255, 255, 255), (0, 0, 0)
    else:
        return (10, 10, 10), (230, 230, 230)

def get_gesture_from_landmarks(landmarks):
    """Classify gesture using MediaPipe landmarks."""
    finger_tips = [8, 12, 16, 20]
    extended = 0
    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            extended += 1
    if extended == 0:
        return "Rock"
    if extended == 4:
        return "Paper"
    if extended == 2:
        return "Scissors"
    return "None"

def skin_mask(frame):
    """Skin detection mask (for fallback)."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower = (0, 133, 77)
    upper = (255, 173, 127)
    mask = cv2.inRange(ycrcb, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return mask

def get_gesture_from_contour(contour, frame=None):
    """Estimate gesture from OpenCV contour."""
    if contour is None or cv2.contourArea(contour) < 2000:
        return "None"
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return "None"
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return "Rock"
    count_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        far = tuple(contour[f][0])
        if d > 10000:
            count_defects += 1
            if frame is not None:
                cv2.circle(frame, far, 5, (0, 0, 255), -1)
    fingers = count_defects + 1 if count_defects > 0 else 0
    if fingers == 0:
        return "Rock"
    if fingers >= 4:
        return "Paper"
    if fingers == 2:
        return "Scissors"
    return "None"

# Webcam
cap = cv2.VideoCapture(0)
start_time = 0
print("\n🖐️ Rock Paper Scissors (Camera Version)\n")
for line in INSTRUCTIONS:
    print(line)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)

    if USE_MEDIAPIPE:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = get_gesture_from_landmarks(hand_landmarks.landmark)
    else:
        mask = skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
            gesture = get_gesture_from_contour(largest, frame)
        else:
            gesture = "None"

    # --- Display gesture info ---
    cv2.putText(frame, f"Your Gesture: {gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Game State Machine
    if state == "WAIT":
        # Determine an area for the instructions box
        ins_x, ins_y = 20, 40
        ins_w, ins_h = 520, max(120, len(INSTRUCTIONS) * 26)

        # choose text and outline color depending on the background
        text_color, outline_color = choose_text_colors(frame, ins_x, ins_y, ins_w, ins_h)

        # draw a semi-transparent background box to improve readability
        overlay = frame.copy()
        box_color = (int(outline_color[0] * 0.15), int(outline_color[1] * 0.15), int(outline_color[2] * 0.15))
        cv2.rectangle(overlay, (ins_x - 8, ins_y - 30), (ins_x + ins_w, ins_y + ins_h), box_color, -1)
        alpha = 0.45
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # draw instruction lines with a thin outline for contrast
        base_y = ins_y
        for i, line in enumerate(INSTRUCTIONS):
            y = base_y + i * 26
            # outline
            cv2.putText(frame, line, (ins_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, outline_color, 3, cv2.LINE_AA)
            # main text
            cv2.putText(frame, line, (ins_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

        # 'Press SPACE' prompt (use contrasting green/orange depending on bg)
        prompt_color, prompt_outline = choose_text_colors(frame, 20, 420 - 30, 300, 40)
        cv2.putText(frame, "Press 'SPACE' to Play!", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, prompt_outline, 3)
        cv2.putText(frame, "Press 'SPACE' to Play!", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

        # Display round result if still within display time
        if round_result and time.time() - result_display_time < 2:
            cv2.putText(frame, round_result, (200, 300),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
    if state == "START":
        cv2.putText(frame, f"Show gesture in {timer}", (180, 200),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)

        if time.time() - start_time >= 1:
            timer -= 1
            start_time = time.time()

        if timer == 0:
            computer_choice = random.choice(choices)

            # Determine round result
            if gesture != "None":
                if gesture == computer_choice:
                    round_result = "Draw!"
                elif (gesture == "Rock" and computer_choice == "Scissors") or \
                     (gesture == "Scissors" and computer_choice == "Paper") or \
                     (gesture == "Paper" and computer_choice == "Rock"):
                    round_result = "You Win!"
                    player_score += 1
                else:
                    round_result = "Computer Wins!"
                    computer_score += 1
            else:
                round_result = "Show Proper Gesture!"

            result_display_time = time.time()  # start timer for displaying result

            # Display computer move
            cv2.putText(frame, round_result, (200, 300), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 0, 0), 3)

            # Display scores
            cv2.putText(frame, f"Player: {player_score}   Computer: {computer_score}",
                        (20, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), 2)
            state = "WAIT"  # go back to waiting state

    # Show frame
    cv2.imshow("Rock Paper Scissors - Camera Game", frame)

    key = cv2.waitKey(1)
    if key == ord(' '):  # start round
        state = "START"
        timer = 3
        start_time = time.time()

    if key == ord('q'):  # Q to Quit
        # Prepare final summary image and show scores + winner for 5 seconds
        # determine frame size
        if 'frame' in locals() and frame is not None:
            h, w = frame.shape[:2]
        else:
            h, w = 480, 640

        summary = np.zeros((h, w, 3), dtype=np.uint8)

        if player_score > computer_score:
            winner_text = "Winner: You!"
        elif player_score < computer_score:
            winner_text = "Winner: Computer!"
        else:
            winner_text = "Winner: Draw!"

        # Draw texts
        cv2.putText(summary, "Final Scores", (int(w*0.05), int(h*0.2)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        cv2.putText(summary, f"Player: {player_score}", (int(w*0.05), int(h*0.45)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(summary, f"Computer: {computer_score}", (int(w*0.05), int(h*0.6)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(summary, winner_text, (int(w*0.05), int(h*0.85)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 180, 255), 3)

        cv2.imshow("Rock Paper Scissors Game - Camera Version", summary)
        # show for 10 seconds (10000 ms)
        cv2.waitKey(10000)
        break

cap.release()
cv2.destroyAllWindows()
