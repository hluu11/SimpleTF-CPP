#define TF_CPP_MIN_VLOG_LEVEL
#include <string>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <thread>
#include <mutex>
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.hpp>

#include <time.h>

#include "Utils.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;
using namespace cv;

int face_detected;
bool gInferReady = true;
// FPS counter
int iFrame = 0;
double fps = 0.;
const int LIMIT_FPS = 30;
const int WAIT_KEY = 1000 / LIMIT_FPS;
tensorflow::TensorShape shape;
std::vector<Tensor> outputs;
// Set input & output nodes names
string inputLayer = "image_tensor:0";
vector<string> outputLayer = { "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0" };
// Load labels map from .pbtxt file
std::map<int, std::string> labelsMap = std::map<int, std::string>();
cv::Mat gFrame, gFrameTF;
std::unique_ptr<tensorflow::Session> session;

std::mutex gLockBuffer;
std::vector<double> gXmin, gYmin, gXmax, gYmax, gScore;
std::vector<double> gXminBuffer, gYminBuffer, gXmaxBuffer, gYmaxBuffer, gScoreBuffer;


void tf_infer()
{
    // Convert mat to tensor
    Tensor tensor = Tensor(tensorflow::DT_FLOAT, shape);
    Status readTensorStatus = readTensorFromMat(gFrameTF, tensor);
    if (!readTensorStatus.ok()) {
        LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
        return;
    }

    double thresholdScore = 0.3;
    double thresholdIOU = 0.5;

    // Run the graph on tensor
    outputs.clear();
    Status runStatus = session->Run({ { inputLayer, tensor } }, outputLayer, {}, &outputs);
    if (!runStatus.ok()) {
        LOG(ERROR) << "Running model failed: " << runStatus;
        return;
    }
    // Extract results from the outputs vector
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float, 3>();

    vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore);
    face_detected = goodIdxs.size();

    // Get bboxes coord
    gXminBuffer.resize(0);
    gYminBuffer.resize(0);
    gXmaxBuffer.resize(0);
    gYmaxBuffer.resize(0);
    GetBoundingBoxesOnImage(scores, classes, boxes, goodIdxs, gXminBuffer, gYminBuffer, gXmaxBuffer, gYmaxBuffer, gScoreBuffer);
    gLockBuffer.lock();
    std::swap(gXmin, gXminBuffer);
    std::swap(gYmin, gYminBuffer);
    std::swap(gXmax, gXmaxBuffer);
    std::swap(gYmax, gYmaxBuffer);
    std::swap(gScore, gScoreBuffer);
    gLockBuffer.unlock();
    gInferReady = true;
}

int main(int argc, char* argv[])
{
    
    // Set dirs variables
    string ROOTDIR = "./";
    string LABELS = "model/labels_map.pbtxt";
    string GRAPH = "model/model.pb";

    string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
    LOG(INFO) << "graphPath:" << graphPath;
    Status loadGraphStatus = loadGraph(graphPath, &session);
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return -1;
    }
    else
        LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;


    
    Status readLabelsMapStatus = readLabelsMapFile(tensorflow::io::JoinPath(ROOTDIR, LABELS), labelsMap);
    if (!readLabelsMapStatus.ok()) {
        LOG(ERROR) << "readLabelsMapFile(): ERROR" << loadGraphStatus;
        return -1;
    }
    else
        LOG(INFO) << "readLabelsMapFile(): labels map loaded with " << labelsMap.size() << " label(s)" << endl;


    time_t start, end;
    time(&start);

    // Start streaming frames from camera
    VideoCapture cap("Ex.mp4");

    shape = tensorflow::TensorShape();
    shape.AddDim(1);
    shape.AddDim((int64)cap.get(CAP_PROP_FRAME_HEIGHT));
    shape.AddDim((int64)cap.get(CAP_PROP_FRAME_WIDTH));
    shape.AddDim(3);
    std::shared_ptr<std::thread> tf_thread;
    tf_thread = nullptr;

    while (cap.isOpened())
    {
        cap >> gFrame;
        if (gFrame.rows == 0)
        {
            break; // endof video
        }

        cvtColor(gFrame, gFrame, COLOR_BGR2RGB);

        if ((iFrame + 1) % (LIMIT_FPS * 10) == 0)
        {
            time(&end);
            fps = double(LIMIT_FPS * 10) / difftime(end, start);
            time(&start);
        }
        if (gInferReady)
        {
            if (tf_thread)
            {
                tf_thread->join();
            }
            gFrame.copyTo(gFrameTF);
            tf_thread.reset(new std::thread(tf_infer));
            gInferReady = false;
        }
        if (iFrame == 0)
        {
            tf_thread->join();
            tf_thread = nullptr;
        }
        iFrame++;

        // draw latest result
        cvtColor(gFrame, gFrame, COLOR_BGR2RGB);
        gLockBuffer.lock();
        for (int i = 0; i < gXmin.size(); i++)
        {
            drawBoundingBoxOnImage(gFrame, gYmin[i], gXmin[i], gYmax[i], gXmax[i], gScore[i], "face", true);
        }
        gLockBuffer.unlock();
        cv::putText(gFrame, to_string(face_detected) + " faces", Point(0, gFrame.rows), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255));
        cv::putText(gFrame, to_string(fps).substr(0,3) + " fps", Point(0, 50), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255));
        
        cv::imshow("Result", gFrame);
        cv::waitKey(WAIT_KEY);
    }
    destroyAllWindows();

    return 0;
}