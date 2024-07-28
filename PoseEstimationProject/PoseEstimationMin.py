import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('PoseVideos/5.mp4')
pTime = 0

# Get the frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate the aspect ratio
aspect_ratio = frame_width / frame_height

# Create a window with the same aspect ratio as the video frame
cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# Get the screen resolution
screen_width = 1920  # Replace with your screen width
screen_height = 1080  # Replace with your screen height

# Calculate the window size while maintaining the aspect ratio
if aspect_ratio * screen_height > screen_width:
    window_width = screen_width
    window_height = int(window_width / aspect_ratio)
else:
    window_height = screen_height
    window_width = int(window_height * aspect_ratio)

cv2.resizeWindow("image", window_width, window_height)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("image", img)
    cv2.waitKey(10)
