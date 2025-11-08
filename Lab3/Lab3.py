import cv2 as cv

def nothing(x):
    pass

def kontury(img):
    cv.namedWindow("image")
    cv.createTrackbar("Threshold", "image", 0, 255, nothing)

    kenrel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)

    while True:
        tresh = cv.getTrackbarPos("Threshold", "image")

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, tresh_img = cv.threshold(gray, tresh, 255, cv.THRESH_BINARY)
        closed = cv.morphologyEx(tresh_img, cv.MORPH_CLOSE, kenrel)
        contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        if contours:
            biggest = max(contours, key=cv.contourArea)
            area = cv.contourArea(biggest)
            perimeter = cv.arcLength(biggest, True)

        img_copy = img.copy()
        cv.drawContours(img_copy, [biggest], -1, (0, 0, 255), 2)
        text = f"Pole: {int(area)} px, Obowd: {int(perimeter)} px"
        cv.putText(img_copy, text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv.imshow("image", img_copy)
        cv.waitKey(1)

kernel_size = (3, 3)
img = cv.imread("image.jpg")
kontury(img)