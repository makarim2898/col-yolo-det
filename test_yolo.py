import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = 0
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        # results = model(frame, conf=0.9, classes='persons',plots=True, verbose = False, stream=True)
        # results = model(frame, max_det=2, conf=0.1, classes = 0)
        results = model(frame, max_det=10, conf=0.1)


        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        for r in results:

            #ganti 0 dengan index class yang dicari, dapat dilihat dengan print(model.names)
            # if 0 in r.boxes.cls: #cari data dari index 0 model di hasil detek
            #     print("ÄDA")
            # else :
            #     print("qwertyuio")

            
        #     print(r.boxes.conf)#untuk ambil data confidence
        #     nilai = r.boxes.conf
            print('#####     my print     #####')
            print(r.__len__())#jumlah object yang di deteksi
            print(r.boxes.cls)#menampilkan index objek yang di deteksi // return value nya adalah array of index object detected
            print(len(r.boxes.cls)) #menampilkan jumlah object yang di deteksi
            data_tensor = r.boxes.cls
            count_index = 0
            count_of_index = (data_tensor == count_index).sum().item()
            print('count 0 in tensor')
            print(count_of_index)
            print('qwertyuiop')
            print('#####     end my print     #####')
            
            


        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()