#!/usr/bin/env python3
import cv2
import paho.mqtt.client as mqtt
import time

BROKER = "localhost"
PORT   = 1883

TOPIC_CAMERA_FRAME = "camera/frame"

# True = affiche la vidéo pour debug
SHOW_DEBUG = True

def main():
    client = mqtt.Client()
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    # ---- CAMERA GAZEBO via GStreamer ----
    # Flux: RTP/H264 sur UDP 5600 (payload=96), comme ton gst-launch qui fonctionne.
    gst_str = (
        "udpsrc port=5600 caps=application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
        "queue ! "
        "rtpjitterbuffer mode=1 latency=50 drop-on-late=true ! "
        "rtph264depay ! h264parse ! avdec_h264 ! "
        "queue ! videoconvert ! "
        "appsink drop=1 max-buffers=1 sync=false"
    )

    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    # Réduit le buffering côté OpenCV (selon backend, peut aider)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[CAM] Impossible d'ouvrir le flux GStreamer (vérifie Gazebo/SITL et le port 5600)")
        client.loop_stop()
        client.disconnect()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[CAM] Plus d'image (flux interrompu ?)")
                # Petite attente puis on retente au lieu de quitter direct (robuste)
                time.sleep(0.2)
                continue

            # Encodage en JPEG
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                client.publish(TOPIC_CAMERA_FRAME, buf.tobytes())

            if SHOW_DEBUG:
                cv2.imshow("camera_publisher", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

            time.sleep(0.03)  # ~30 Hz

    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()

