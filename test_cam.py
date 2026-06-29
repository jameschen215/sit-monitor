import cv2 as cv
import sys


def main():
    cap = cv.VideoCapture(
        "rtsp://admin:TUHXOF@192.168.0.109:554/h264/ch1/main/av_stream"
    )

    if not cap.isOpened():
        print("Error: cannot open camera")
        sys.exit(1)

    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera connected! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: cannot read frame")
            break

        frame = cv.resize(frame, (640, 360))  # 16:9 比例
        cv.imshow("Action 4 Test", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
