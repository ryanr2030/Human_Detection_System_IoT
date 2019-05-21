//
//  main.cpp
//  human-detector

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iterator>
#include <ctime>
#include <opencv2/videoio.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/core/utils/filesystem.hpp"







std::string keys =
"{ help  h     | | Print help message. }"
"{ haar  Haar  | | use haar cascading }"
"{ dnn  DNN  | | use deep learning method}"
"{ hog  Hog  | | use deep learning method}"
"{ video | | save the video to a file }"
"{ display  Display | | display the processed frame to the screen}"
"{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
"{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
"{ device      |  0 | camera device number. }"
"{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
"{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
"{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
"{ thr         | .5 | Confidence threshold. }"
"{ nms         | .4 | Non-maximum suppression threshold. }"
"{ backend     |  0 | Choose one of computation backends: "
"0: automatically (by default), "
"1: Halide language (http://halide-lang.org/), "
"2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
"3: OpenCV implementation }"
"{ target      | 0 | Choose one of target computation devices: "
"0: CPU target (by default), "
"1: OpenCL, "
"2: OpenCL fp16 (half-float precision), "
"3: VPU }";


using namespace cv;
using namespace dnn;

float confThreshold, nmsThreshold;
std::vector<std::string> classes;

std::vector<std::string> postprocess(Mat& frame, const std::vector<Mat>& out, Net& net, int framecount, CommandLineParser parser);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

void callback(int pos, void* userdata);
std::vector <std::string> haarDetectAndDisplay( Mat& frame, int framecount, CommandLineParser parser);
std::vector <std::string> hogDetectAndDisplay(Mat& frame, HOGDescriptor& hog, Size win_stride, int framecount, CommandLineParser parser);


std::string genArgument(const std::string& argName, const std::string& help,
                        const std::string& modelName, const std::string& zooFile,
                        char key = ' ', std::string defaultVal = "");

std::string genPreprocArguments(const std::string& modelName, const std::string& zooFile);

std::string findFile(const std::string& filename);
std::vector<std::string> getOutputsNames(const Net& net);


String haar_human_cascade_name = "/home/pi/Desktop/project_master/cpp/input_files/haar/haarcascade_fullbody.xml";
String hog_human_cascade_name = "/classifiers/hogcascade_pedestrians.xml";
CascadeClassifier human_cascade;
int nlevels;
bool make_gray;
double scale_hog;
double resize_scale;
int win_width;
int win_stride_width, win_stride_height;
int gr_threshold;
double hit_threshold;
bool gamma_corr;




int main(int argc, char** argv)
{
    VideoWriter writer;
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)
    double fps_vid = 10, fps=0;                          // framerate of the created video stream
    std::string outvid="/home/pi/Desktop/project_master/cpp/output_files/test.avi";
    CommandLineParser parser(argc, argv, keys);
    std::vector<std::string> detections;
    //Hog Locals
    win_width = 48;
    win_stride_width = 8;
    win_stride_height = 8;
    gr_threshold = 2;
    nlevels = 13;
    hit_threshold = 1;
    scale_hog = 1.059;
    gamma_corr = true;
    Size win_size(win_width, win_width * 2);
    Size win_stride(win_stride_width, win_stride_height);
    HOGDescriptor hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1, HOGDescriptor::L2Hys, 0.2, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);

    //DNN Locals
    const std::string modelName = parser.get<String>("@alias");
    const std::string zooFile = parser.get<String>("zoo");
    std::vector<std::string> outNames;
    Net net;
    float scale;
    bool swapRB;
    Scalar mean;
    int inpWidth;
    int inpHeight;
    keys += genPreprocArguments(modelName, zooFile);
    time_t absStart, start, stop;

    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run object detection deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    else if(parser.has("haar") || parser.has("Haar")){
        if(!human_cascade.load(haar_human_cascade_name)){
            printf("--ERROR LOADING HAAR CLASSIFIER\n");
            return -1;
        }
    }
    else if(parser.has("hog")){
        hog.setSVMDetector( HOGDescriptor::getDaimlerPeopleDetector() );
    }
    else if(parser.has("dnn")){
        confThreshold = parser.get<float>("thr");
        nmsThreshold = parser.get<float>("nms");
        scale = parser.get<float>("scale");
        mean = parser.get<Scalar>("mean");
        swapRB = parser.get<bool>("rgb");
        inpWidth = parser.get<int>("width");
        inpHeight = parser.get<int>("height");
        CV_Assert(parser.has("model"));
        std::string modelPath = findFile(parser.get<String>("model"));
        std::string configPath = findFile(parser.get<String>("config"));
        
        // Open file with classes names.
        if (parser.has("classes"))
        {
            std::string file = parser.get<String>("classes");
            std::ifstream ifs(file.c_str());
            if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
            std::string line;
            while (std::getline(ifs, line))
            {
                classes.push_back(line);
            }
        }
        
        // Load a model.
        net = readNet(modelPath, configPath, parser.get<String>("framework"));
        net.setPreferableBackend(parser.get<int>("backend"));
        net.setPreferableTarget(parser.get<int>("target"));
        outNames = net.getUnconnectedOutLayersNames();
    }
    static const std::string kWinName = "Deep learning object detection in OpenCV";
    if(parser.has("display")){

        // Create a window
        namedWindow(kWinName, WINDOW_NORMAL);
        int initialConf = (int)(confThreshold * 100);
        //createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback);

    }
    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    
    //

    if (parser.has("input"))
        cap.open("/home/pi/Downloads/inputvid2.mp4");
    else
        cap.open(parser.get<int>("device"));
    int fcount=1;
    start=stop=time(&absStart);
    time_t last_upload;
    last_upload=(start);
    // Process frames.
    Mat frame, blob;
    UMat frame_img;
    std::vector<std::string> temp;

     
    while (waitKey(1) < 0)
    {
        time(&start);
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }
        bool isColor = (frame.type() == CV_8UC3);
        //open video writer
        if(parser.has("video")){
            if (!writer.isOpened()) {
                writer.open(outvid, codec, fps_vid, frame.size(), isColor);
                std::cout<<"WRITER OPENED FOR: "<<outvid<<"\n";
                if (!writer.isOpened()) {
                    std::cerr << "Could not open the output video file for write\n";
                    return -1;
                }
            }
        }
        
        if(parser.has("dnn")){
            // Create a 4D blob from a frame.
            Size inpSize(inpWidth > 0 ? inpWidth : frame.cols,
                         inpHeight > 0 ? inpHeight : frame.rows);
            blobFromImage(frame, blob, scale, inpSize, mean, swapRB, false);
            
            // Run a model.
            net.setInput(blob);
            if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
            {
                resize(frame, frame, inpSize);
                Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
                net.setInput(imInfo, "im_info");
            }
            std::vector<Mat> outs;
            net.forward(outs, outNames);
            
           
            temp=postprocess(frame, outs, net, fcount, parser);
       

            detections.insert(detections.end(), temp.begin(), temp.end());
            temp.clear();
        }
        //append strings in haar, hog,yolo functions
        else if(parser.has("haar")){
            if(!frame.empty())
                temp=haarDetectAndDisplay(frame, fcount, parser);
                detections.insert(detections.end(), temp.begin(), temp.end());
                temp.clear();
        }
        else if(parser.has("hog")){
            if(!frame.empty()){
                temp=hogDetectAndDisplay(frame, hog, win_stride, fcount, parser);
                detections.insert(detections.end(), temp.begin(), temp.end());
                temp.clear();
               }
            
        }
        // Put efficiency information.
        if(parser.has("dnn")){
            std::vector<double> layersTimes;
            double freq = getTickFrequency() / 1000;
            double t = net.getPerfProfile(layersTimes) / freq;
            fps=1/(t*.001);
            std::string label = format("Inference time: %.2f ms FPS: %.2f", t, fps);
            putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        }
        else{
            time(&stop);
            double elapsed=(stop-start)*1000;
            double absElapsed=stop-absStart;
            fps=fcount/absElapsed;
            std::string label = format("FPS: %.2f", fps);
            putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        }
        
        //Write a time delayed packet (max files size 256kb)
        time_t end_time, elapsed_time;
        time(&end_time);
        elapsed_time = end_time - last_upload;
        if(elapsed_time>=0){
            std::ofstream detectionfile("/home/pi/Desktop/project_master/cpp/output_files/data.json");
            detectionfile<<"{\"detections\": { \"humans\":[";
            //std::ostream_iterator<std::string> output_iterator(detectionfile,"\n");
            //std::copy(detections.begin(), detec+tions.end(), output_iterator);
            for(int i=0; i<detections.size(); i++){
                detectionfile<<detections.at(i);
                std::cout<<detections.at(i);
            }
            std::ostringstream strs;
            strs << fps;
            std::string FPS = strs.str();
            detectionfile<<"] , \"fps\": \""+FPS+"\"}}";
            detections.clear();
            detectionfile.close();
            std::cout<<"\n__________________PACKET WRITTEN FOR TRANSMISSION__________________\n";
        }
        if(parser.has("display")){
            imshow(kWinName, frame);
        }
        if(parser.has("video")){
            if(writer.isOpened()){
                std::cout<<"WRITING TO FILE"<<std::endl;
                writer.write(frame);
            }            
        }
        fcount++;
    
    }
    return 0;
}

std::vector<std::string> postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, int framecount, CommandLineParser parser)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;
    
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++)
        {
            float* data = (float*)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left   = (int)data[i + 3];
                    int top    = (int)data[i + 4];
                    int right  = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width * height <= 1)
                    {
                        left   = (int)(data[i + 3] * frame.cols);
                        top    = (int)(data[i + 4] * frame.rows);
                        right  = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }
                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back(Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
    }
    else if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    else{
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);}
    

    std:: vector<std::string> detections, test;
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        if(classIds[idx]==0){
            Rect box = boxes[idx];
            drawPred(classIds[idx], confidences[idx], box.x, box.y,
                     box.x + box.width, box.y + box.height, frame);
            int c_x = (box.x+box.width)/2;
            int c_y = (box.y+box.height)/2;
            std::string holder = std::to_string(framecount)+" "+std::to_string(i + 1)+" "+std::to_string(box.x)+" "+std::to_string(box.y)+" "+std::to_string(box.x + box.width)+
            " "+std::to_string(box.y + box.height)+"\n";
            std::string det = "{\"frame\": \"" + std::to_string(framecount) + "\", \"num_in_frame\": \"" + std::to_string(i + 1) + "\", \"x\": \"" 
                + std::to_string(c_x) + "\", \"y\": \"" + std::to_string(c_y) + "\"},\n";
            detections.push_back (det);
            test.push_back (holder);

        }
    
    }
    if(detections.size()>=1){
        std::string last= detections[detections.size()-1];
        last.pop_back();
        last.pop_back();
        detections[detections.size()-1]=last;
    }
    if(parser.has("input")){
        std::ofstream detectfile;
        detectfile.open("/home/pi/Desktop/project_master/cpp/output_files/detections.txt", std::ios_base::app);
        if(test.size()== 0){
            detectfile<<std::to_string(framecount)+"\n";
        }
        else{
            for( int i = 0; i<test.size(); i++){
                detectfile<<test.at(i);
            }
        }
        detectfile.close();
    }

    return detections;
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
    
    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }
    
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());

}

//method to draw prediction for Haar
void drawPredHaar(int left, int top, int right, int bottom, Mat& frame)
{
    std::string label = "Person:";
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

void callback(int pos, void*)
{
    confThreshold = pos * 0.01f;
}





std::string genArgument(const std::string& argName, const std::string& help,
                        const std::string& modelName, const std::string& zooFile,
                        char key, std::string defaultVal)
{
    if (!modelName.empty())
    {
        FileStorage fs(zooFile, FileStorage::READ);
        if (fs.isOpened())
        {
            FileNode node = fs[modelName];
            if (!node.empty())
            {
                FileNode value = node[argName];
                if (!value.empty())
                {
                    if (value.isReal())
                    defaultVal = format("%f", (float)value);
                    else if (value.isString())
                    defaultVal = (std::string)value;
                    else if (value.isInt())
                    defaultVal = format("%d", (int)value);
                    else if (value.isSeq())
                    {
                        for (size_t i = 0; i < value.size(); ++i)
                        {
                            FileNode v = value[(int)i];
                            if (v.isInt())
                            defaultVal += format("%d ", (int)v);
                            else if (v.isReal())
                            defaultVal += format("%f ", (float)v);
                            else
                            CV_Error(Error::StsNotImplemented, "Unexpected value format");
                        }
                    }
                    else
                    CV_Error(Error::StsNotImplemented, "Unexpected field format");
                }
            }
        }
    }
    return "{ " + argName + " " + key + " | " + defaultVal + " | " + help + " }";
}

std::string findFile(const std::string& filename)
{
    if (filename.empty() || utils::fs::exists(filename))
    return filename;
    
    const char* extraPaths[] = {getenv("OPENCV_DNN_TEST_DATA_PATH"),
        getenv("OPENCV_TEST_DATA_PATH")};
    for (int i = 0; i < 2; ++i)
    {
        if (extraPaths[i] == NULL)
        continue;
        std::string absPath = utils::fs::join(extraPaths[i], utils::fs::join("dnn", filename));
        if (utils::fs::exists(absPath))
        return absPath;
    }
    CV_Error(Error::StsObjectNotFound, "File " + filename + " not found! "
             "Please specify a path to /opencv_extra/testdata in OPENCV_DNN_TEST_DATA_PATH "
             "environment variable or pass a full path to model.");
}

std::string genPreprocArguments(const std::string& modelName, const std::string& zooFile)
{
    return genArgument("model", "Path to a binary file of model contains trained weights. "
                       "It could be a file with extensions .caffemodel (Caffe), "
                       ".pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet), .bin (OpenVINO).",
                       modelName, zooFile, 'm') +
    genArgument("config", "Path to a text file of model contains network configuration. "
                "It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet), .xml (OpenVINO).",
                modelName, zooFile, 'c') +
    genArgument("mean", "Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces.",
                modelName, zooFile) +
    genArgument("scale", "Preprocess input image by multiplying on a scale factor.",
                modelName, zooFile, ' ', "1.0") +
    genArgument("width", "Preprocess input image by resizing to a specific width.",
                modelName, zooFile, ' ', "-1") +
    genArgument("height", "Preprocess input image by resizing to a specific height.",
                modelName, zooFile, ' ', "-1") +
    genArgument("rgb", "Indicate that model works with RGB input images instead BGR ones.",
                modelName, zooFile);
}
/*HAAR DETECTION */
/** @function detectAndDisplay */
std::vector <std::string> haarDetectAndDisplay(Mat& frame, int framecount, CommandLineParser parser)
{
    std::vector<String> detections, test;
    std::vector<Rect> humans;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    //-- Detect Humans
    human_cascade.detectMultiScale( frame_gray, humans, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(20, 20) );
    for(size_t i=0; i<humans.size();i++){
        drawPredHaar(humans[i].x, humans[i].y, humans[i].x+humans[i].width, humans[i].y+humans[i].height, frame);
        int c_x = (humans[i].x+humans[i].width)/2;
        int c_y = (humans[i].y+humans[i].height)/2;
        std::string holder = std::to_string(framecount)+" "+std::to_string(i + 1)+" "+std::to_string(humans[i].x)+" "+std::to_string(humans[i].y)+" "+std::to_string(humans[i].x + humans[i].width)+
            " "+std::to_string(humans[i].y + humans[i].height)+"\n";
        if(i==humans.size()-1){
            std::string det = "{\"frame\": \"" + std::to_string(framecount) + "\", \"num_in_frame\": \"" + std::to_string(i + 1) + "\", \"x\": \"" 
            + std::to_string(c_x) + "\", \"y\": \"" + std::to_string(c_y) + "\"}\n";
            detections.push_back (det);
        }
        else{
            std::string det = "{\"frame\": \"" + std::to_string(framecount) + "\", \"num_in_frame\": \"" + std::to_string(i + 1) + "\", \"x\": \"" 
            + std::to_string(c_x) + "\", \"y\": \"" + std::to_string(c_y) + "\"},\n";
            detections.push_back (det);
        }
        test.push_back (holder);
    }
     
    if(parser.has("input")){
        std::ofstream detectfile;
        detectfile.open("/home/pi/Desktop/project_master/cpp/output_files/detections.txt", std::ios_base::app);
        if(test.size()== 0){
            detectfile<<std::to_string(framecount)+"\n";
        }
        else{
            for( int i = 0; i<test.size(); i++){
                detectfile<<test.at(i);
            }
        }
        detectfile.close();
    }
    return detections;
}

/**HOG DETECTION*/
std::vector<std::string> hogDetectAndDisplay(Mat& frame, HOGDescriptor& hog, Size win_stride, int framecount, CommandLineParser parser){
    
    std::vector<std::string> detections, test;
    UMat frm;
    cvtColor(frame, frm, COLOR_BGR2GRAY );
    hog.nlevels=nlevels;
    std::vector<Rect> human;
    hog.detectMultiScale(frm, human, hit_threshold, win_stride, Size(0,0), scale_hog,gr_threshold);
    for (size_t i = 0; i < human.size(); i++)
    {
        Rect r = human[i];
        rectangle(frame, r.tl(), r.br(), Scalar(0, 255, 0), 3);
        int c_x = (human[i].x+human[i].width)/2;
        int c_y = (human[i].y+human[i].height)/2;
        std::string holder = std::to_string(framecount)+" "+std::to_string(i + 1)+" "+std::to_string(human[i].x)+" "+std::to_string(human[i].y)+" "+std::to_string(human[i].x + human[i].width)+
            " "+std::to_string(human[i].y + human[i].height)+"\n";
        if(i==human.size()-1){
            std::string det = "{\"frame\": \"" + std::to_string(framecount) + "\", \"num_in_frame\": \"" + std::to_string(i + 1) + "\", \"x\": \"" 
            + std::to_string(c_x) + "\", \"y\": \"" + std::to_string(c_y) + "\"}\n";
            detections.push_back (det);
        }
        else{
            std::string det = "{\"frame\": \"" + std::to_string(framecount) + "\", \"num_in_frame\": \"" + std::to_string(i + 1) + "\", \"x\": \"" 
            + std::to_string(c_x) + "\", \"y\": \"" + std::to_string(c_y) + "\"},\n";
            detections.push_back (det);
        }
        test.push_back(holder);
    }
    
    if(parser.has("input")){
        std::ofstream detectfile;
        detectfile.open("/home/pi/Desktop/project_master/cpp/output_files/detections.txt", std::ios_base::app);
        if(test.size()== 0){
            detectfile<<std::to_string(framecount)+"\n";
        }
        else{
            for( int i = 0; i<test.size(); i++){
                detectfile<<test.at(i);
            }
        }
        detectfile.close();
    }
    return detections;
    
}

//Raspberry Pi Helpers
static cv::Mat GetImageFromCamera(cv::VideoCapture& camera)
{
    cv::Mat frame;
    camera >> frame;
    return frame;
}

