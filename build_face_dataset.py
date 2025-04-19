import cv2

count = 0

cap = cv2.VideoCapture(0)

while cap.isOpened(): 
    ret, frame = cap.read()
    cv2.imwrite(f"/Users/thanadonxmac/Documents/Python/Project/dataset/นางสาวปนัดดา อุตคุต/ปนัดดา_{count}.jpg",frame)
    count += 1
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    
cv2.destroyAllWindows()
cap.release()