from ultralytics import YOLO
import cv2

def main():
    # TESTE 
    # model = YOLO("best.pt")

    # TESTE  2
    model = YOLO("models/yolov8n.pt")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.50, device="cpu")

        annotated = results[0].plot()

        cv2.imshow("YOLO Webcam", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()