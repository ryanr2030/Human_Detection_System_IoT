#!/bin/bash
DET_PATH="/home/pi/Desktop/project_master/cpp/build"
python /home/pi/Desktop/project_master/cpp/transmit.py &

if [ "$1" = "yolo" ]
then
    if [ "$2" = "video" ] || [ "$3" = "video" ]; then
        if [ "$2" = "display" ] || [ "$3" = "display"]; then
            ($DET_PATH/detector --dnn --config=/home/pi/Desktop/project_master/cpp/input_files/yolo/cfg/yolov3-tiny.cfg \
            --model=/home/pi/Desktop/project_master/cpp/input_files/yolo/weights/yolov3-tiny.weights \
            --classes=/home/pi/Desktop/project_master/cpp/input_files/yolo/names/coco.names --width=416 \
            --height=416 --scale=0.00392 --rgb --thr=.5 --nms=.3  --video --display)
        else
            ($DET_PATH/detector --dnn --config=/home/pi/Desktop/project_master/cpp/input_files/yolo/cfg/yolov3-tiny.cfg \
            --model=/home/pi/Desktop/project_master/cpp/input_files/yolo/weights/yolov3-tiny.weights \
            --classes=/home/pi/Desktop/project_master/cpp/input_files/yolo/names/coco.names --width=416 \
            --height=416 --scale=0.00392 --rgb --thr=.5 --nms=.3  --video)
        fi
    elif [ "$2" = "display" ] || [ "$3" = "display"]; then
            ($DET_PATH/detector --dnn --config=/home/pi/Desktop/project_master/cpp/input_files/yolo/cfg/yolov3-tiny.cfg \
            --model=/home/pi/Desktop/project_master/cpp/input_files/yolo/weights/yolov3-tiny.weights \
            --classes=/home/pi/Desktop/project_master/cpp/input_files/yolo/names/coco.names --width=416 \
            --height=416 --scale=0.00392 --rgb --thr=.5 --nms=.3 --display)
    else
            ($DET_PATH/detector --dnn --config=/home/pi/Desktop/project_master/cpp/input_files/yolo/cfg/yolov3-tiny.cfg \
            --model=/home/pi/Desktop/project_master/cpp/input_files/yolo/weights/yolov3-tiny.weights \
            --classes=/home/pi/Desktop/project_master/cpp/input_files/yolo/names/coco.names --width=416 \
            --height=416 --scale=0.00392 --rgb --thr=.5 --nms=.3)
    fi
elif [ "$1" = "hog" ]
then
    if [ "$2" = "video" ] || [ "$3" = "video" ]; then
        if [ "$2" = "display" ] || [ "$3" = "display"]; then
            ($DET_PATH/detector --hog --video --display)
        else
            ($DET_PATH/detector --hog --video)
        fi
    elif [ "$2" = "display" ] || [ "$3" = "display"]; then
        ($DET_PATH/detector --hog --display)
    else
        $DET_PATH/detector --hog
    fi

elif [ "$1" = "haar" ]
then
    if [ "$2" = "video" ] || [ "$3" = "video" ]; then
        if [ "$2" = "display" ] || [ "$3" = "display"]; then
            (xterm -e $DET_PATH/detector --haar --video --display)
        else
            ($DET_PATH/detector --haar --video)
        fi
    elif [ "$2" = "display" ] || [ "$3" = "display"]; then
        ($DET_PATH/detector --haar --display)

    else
        $DET_PATH/detector --haar      
    fi
else
    echo "Invalid option: please specify hog, haar, or yolo"
    echo "To write the output video specify: video"
    echo "To display the output to the screen specify: display"
fi


