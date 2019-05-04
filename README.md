#Low Cost Real Time Human Detection Using IoT communication

This project uses a singleboard computer (Raspberry Pi 3b+) to detect humans in a videostream.
The detections are uploaded to an Azure IoT routed to a node JS webserver. The node JS webserver post processes the data then emits to clients in realtime on a semi-static webpage using sockets.
Detection Methods:
  Haar,
  Hog,
  Yolov3,
  and TinyYoloV3
