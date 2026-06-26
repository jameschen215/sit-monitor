import cv2 as cv
import sys


def main():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: cannot open camera")
        sys.exit(1)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera connected! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: cannot read frame")
            break

        cv.imshow("Action 4 Test", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
