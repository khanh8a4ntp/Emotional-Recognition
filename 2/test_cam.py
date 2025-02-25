import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print ("ko mo dc")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print ("ko nhan frame")
        break
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

