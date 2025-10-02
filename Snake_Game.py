"""
Gesture-Controlled Snake Game
- Requirements: opencv-python, mediapipe, numpy
- Run: python gesture_snake.py
Controls (gesture):
 - Pointing direction (index finger direction) -> steer snake
 - Fist (all fingers down) -> pause/unpause
Keyboard:
 - 'r' -> restart
 - 'Esc' -> quit
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import random

# ---------------- Settings ----------------
CAM_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

GRID_SIZE = 20              # pixel size of each grid cell
COLS = FRAME_WIDTH // GRID_SIZE
ROWS = FRAME_HEIGHT // GRID_SIZE

SNAKE_COLOR = (0, 255, 0)
FOOD_COLOR = (0, 0, 255)
BG_COLOR = (20, 20, 20)
TEXT_COLOR = (200, 200, 200)

STEP_TIME = 0.12            # seconds per snake move (game speed)
ANGLE_THRESHOLD = 30        # degrees tolerance for cardinal directions
DIRECTION_DEBOUNCE = 0.12   # seconds between allowed direction changes
# ------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- Helper functions ---
def angle_deg(v):
    """Return angle in degrees for 2D vector v (x,y) with 0 degrees pointing right, CCW positive."""
    x, y = v
    ang = np.degrees(np.arctan2(-y, x))  # negative y because screen y grows downward
    return ang % 360

def map_angle_to_dir(a):
    """Map angle (deg) to one of four directions (dx, dy)"""
    # RIGHT: -45..45 -> 315..360 and 0..45
    # UP: 45..135
    # LEFT: 135..225
    # DOWN: 225..315
    if (a <= 45) or (a >= 315):
        return (1, 0)   # RIGHT
    if 45 < a <= 135:
        return (0, -1)  # UP
    if 135 < a <= 225:
        return (-1, 0)  # LEFT
    return (0, 1)       # DOWN

def inside_grid(pt):
    x, y = pt
    return 0 <= x < COLS and 0 <= y < ROWS

def spawn_food(snake):
    while True:
        fx = random.randrange(0, COLS)
        fy = random.randrange(0, ROWS)
        if (fx, fy) not in snake:
            return (fx, fy)

# --- Initialize game state ---
def init_game():
    midx = COLS // 2
    midy = ROWS // 2
    snake = [(midx, midy), (midx-1, midy), (midx-2, midy)]
    direction = (1, 0)
    food = spawn_food(snake)
    score = 0
    return snake, direction, food, score

snake, direction, food, score = init_game()
last_move_time = time.time()
last_dir_change = 0
paused = False
game_over = False

cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Starting Gesture-Controlled Snake")
print(" - Point with your index finger to steer.")
print(" - Make a fist to pause/unpause.")
print(" - Press 'r' to restart, Esc to quit.")

while True:
    ret, cam_frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(cam_frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    idx_tip_px = None
    idx_pip_px = None
    fingers_up = [0,0,0,0]  # index,middle,ring,pinky default

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        h, w = frame.shape[:2]
        def to_px(i): return (int(lm[i].x * w), int(lm[i].y * h))

        idx_tip_px = to_px(8)
        idx_pip_px = to_px(6)
        # detect fingers up (index,middle,ring,pinky)
        tip_ids = [8, 12, 16, 20]
        for i, tip in enumerate(tip_ids):
            fingers_up[i] = 1 if lm[tip].y < lm[tip-2].y else 0

        mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        # visualize index vector
        cv2.circle(frame, idx_tip_px, 6, (255,255,255), -1)
        cv2.line(frame, idx_pip_px, idx_tip_px, (180,180,255), 2)

    # ---- Gesture -> direction mapping ----
    now = time.time()
    if idx_tip_px and idx_pip_px:
        vec = (idx_tip_px[0] - idx_pip_px[0], idx_tip_px[1] - idx_pip_px[1])
        vnorm = np.hypot(vec[0], vec[1])
        if vnorm > 10:  # significant pointing
            ang = angle_deg(vec)
            new_dir = map_angle_to_dir(ang)
            # Prevent reversing directly
            if now - last_dir_change > DIRECTION_DEBOUNCE:
                if (new_dir[0] * -1, new_dir[1] * -1) != direction:
                    direction = new_dir
                    last_dir_change = now

    # Pause/unpause if fist (all fingers down)
    if res.multi_hand_landmarks and fingers_up == [0,0,0,0]:
        # toggle pause only when first detected (avoid toggling many times)
        if not paused and (now - last_move_time) > 0.15:
            paused = True
            # small sleep to avoid instant unpause
            time.sleep(0.15)
    elif res.multi_hand_landmarks:
        # if open hand detected, resume
        if paused and (sum(fingers_up) > 0):
            paused = False
            time.sleep(0.05)

    # ---- Game update tick ----
    if not paused and not game_over and (now - last_move_time) >= STEP_TIME:
        last_move_time = now
        head = snake[0]
        new_head = (head[0] + direction[0], head[1] + direction[1])

        # check collisions
        if (not inside_grid(new_head)) or (new_head in snake):
            game_over = True
        else:
            snake.insert(0, new_head)
            if new_head == food:
                score += 1
                food = spawn_food(snake)
            else:
                snake.pop()

    # ---- Render the game to an image ----
    # Create background
    game_img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    game_img[:] = BG_COLOR

    # Draw grid (optional light)
    for x in range(0, FRAME_WIDTH, GRID_SIZE):
        cv2.line(game_img, (x,0), (x,FRAME_HEIGHT), (15,15,15), 1)
    for y in range(0, FRAME_HEIGHT, GRID_SIZE):
        cv2.line(game_img, (0,y), (FRAME_WIDTH,y), (15,15,15), 1)

    # Draw food
    fx, fy = food
    cv2.rectangle(game_img,
                  (fx*GRID_SIZE, fy*GRID_SIZE),
                  (fx*GRID_SIZE + GRID_SIZE, fy*GRID_SIZE + GRID_SIZE),
                  FOOD_COLOR, -1)

    # Draw snake
    for i, (sx, sy) in enumerate(snake):
        pt1 = (sx*GRID_SIZE, sy*GRID_SIZE)
        pt2 = (sx*GRID_SIZE + GRID_SIZE, sy*GRID_SIZE + GRID_SIZE)
        if i == 0:
            cv2.rectangle(game_img, pt1, pt2, (0,200,0), -1)  # head slightly brighter
        else:
            cv2.rectangle(game_img, pt1, pt2, SNAKE_COLOR, -1)

    # Overlay small camera view in the corner (so user sees hand)
    cam_small = cv2.resize(frame, (160, 120))
    gs_h, gs_w = game_img.shape[:2]
    game_img[5:5+cam_small.shape[0], 5:5+cam_small.shape[1]] = cam_small

    # Draw HUD
    cv2.putText(game_img, f"Score: {score}", (FRAME_WIDTH-160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
    cv2.putText(game_img, f"{'PAUSED' if paused else ''}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

    if game_over:
        cv2.putText(game_img, "GAME OVER - Press 'r' to restart", (50, FRAME_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Gesture Snake", game_img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('r'):
        snake, direction, food, score = init_game()
        game_over = False
        paused = False

cap.release()
cv2.destroyAllWindows()
