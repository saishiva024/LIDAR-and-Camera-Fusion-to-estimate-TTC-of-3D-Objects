
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    size_t numImages = imgEndIndex - imgStartIndex;

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */
    vector<AlgoCharacteristics> algoCharacteristics;
    InitAlgoCombinations(algoCharacteristics);

    for(auto &algoCharacteristic : algoCharacteristics)
    {
        dataBuffer.clear();

        for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
        {
            /* LOAD IMAGE INTO BUFFER */

            // assemble filenames for current index
            ostringstream imgNumber;
            imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
            string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

            // load image from file 
            cv::Mat img = cv::imread(imgFullFilename);

            // push image into data frame buffer
            DataFrame frame;
            frame.cameraImg = img;

            // if(dataBuffer.size() == dataBufferSize)
            // {
            //     dataBuffer.erase(dataBuffer.begin());
            // }

            if(dataBuffer.size() < dataBufferSize)
            {
                dataBuffer.push_back(frame);
            }
            else
            {
                dataBuffer.erase(dataBuffer.begin());
                dataBuffer.push_back(frame);
            }
            
            
            //dataBuffer.push_back(frame);

            cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


            /* DETECT & CLASSIFY OBJECTS */

            float confThreshold = 0.2;
            float nmsThreshold = 0.4;        
            detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold,
                        nmsThreshold,yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

            cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


            /* CROP LIDAR POINTS */

            // load 3D Lidar points from file
            string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
            std::vector<LidarPoint> lidarPoints;
            loadLidarFromFile(lidarPoints, lidarFullFilename);

            // remove Lidar points based on distance properties
            float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
            cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
        
            (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

            cout << "#3 : CROP LIDAR POINTS done" << endl;


            /* CLUSTER LIDAR POINT CLOUD */

            // associate Lidar points with camera-based ROI
            // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
            float shrinkFactor = 0.10; 
            clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints,
                                shrinkFactor, P_rect_00, R_rect_00, RT);

            // Visualize 3D objects
            bVis = false;
            if(bVis)
            {
                cout<<"Visualize 3D Objects"<<endl;
                show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
            }
            bVis = false;

            cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
            
            
            // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
            //continue; // skips directly to the next image without processing what comes beneath

            /* DETECT IMAGE KEYPOINTS */

            // convert current image to grayscale
            cv::Mat imgGray;
            cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

            // extract 2D keypoints from current image
            vector<cv::KeyPoint> keypoints; // create empty feature list for current image
            string detectorType = algoCharacteristic.detector;//HARRIS, SHITOMASI, FAST, BRISK, ORB, AKAZE, SIFT

            bVis = false;
            if (detectorType.compare("HARRIS") == 0)
            {
                detKeypointsHarris(keypoints, imgGray, algoCharacteristic.detectorElapsedTime.at(imgIndex), bVis);
            }
            else if (detectorType.compare("SHITOMASI") == 0)
            {
                detKeypointsShiTomasi(keypoints, imgGray, algoCharacteristic.detectorElapsedTime.at(imgIndex), bVis);
            }
            else
            {
                detKeypointsModern(keypoints, imgGray, detectorType, algoCharacteristic.detectorElapsedTime.at(imgIndex), bVis);
            }

            bVis = false;

            algoCharacteristic.numKpts.at(imgIndex) = keypoints.size();

            bool bFocusOnVehicle = true;
            vector<cv::KeyPoint> precedingVehicleKeypoints;
            
            cv::Rect vehicleRect(535, 180, 180, 150);
            if (bFocusOnVehicle)
            {
                for (auto itr = keypoints.begin(); itr != keypoints.end(); ++itr)
                {
                    if (vehicleRect.contains((*itr).pt))
                    {
                        precedingVehicleKeypoints.push_back(*itr);
                    }
                }
                keypoints = precedingVehicleKeypoints;

                algoCharacteristic.numKptsVehicle.at(imgIndex) = keypoints.size(); 
            }

            // optional : limit number of keypoints (helpful for debugging and learning)
            bool bLimitKpts = false;
            if (bLimitKpts)
            {
                int maxKeypoints = 50;

                if (detectorType.compare("SHITOMASI") == 0)
                { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                    keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                }
                cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                cout << " NOTE: Keypoints have been limited!" << endl;
            }

            // push keypoints and descriptor for current frame to end of data buffer
            (dataBuffer.end() - 1)->keypoints = keypoints;

            cout << "#5 : DETECT KEYPOINTS done" << endl;


            /* EXTRACT KEYPOINT DESCRIPTORS */

            cv::Mat descriptors;

            string descriptorType = algoCharacteristic.descriptor; // BRIEF, ORB, FREAK, AKAZE, SIFT

            descKeypoints((dataBuffer.end() - 1)->keypoints,(dataBuffer.end() - 1)->cameraImg, 
                           descriptors, descriptorType, algoCharacteristic.descriptorElapsedTime.at(imgIndex));

            algoCharacteristic.numDescriptors.at(imgIndex) = descriptors.size().width * descriptors.size().height;

            // push descriptors for current frame to end of data buffer
            (dataBuffer.end() - 1)->descriptors = descriptors;

            cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


            if (dataBuffer.size() > 1) // wait until at least two images have been processed
            {

                /* MATCH KEYPOINT DESCRIPTORS */
                vector<cv::DMatch> matches;
                string matcherType = algoCharacteristic.matcher;        // MAT_BF, MAT_FLANN
                string selectorType = algoCharacteristic.selector;       // SEL_NN, SEL_KNN

                // DES_BINARY, DES_HOG
                string descriptorType = (algoCharacteristic.descriptor.compare("SIFT")==0 ) ? "DES_HOG" : "DES_BINARY";

                matchDescriptors((dataBuffer.end() - 2)->keypoints,
                                 (dataBuffer.end() - 1)->keypoints,
                                 (dataBuffer.end() - 2)->descriptors,
                                 (dataBuffer.end() - 1)->descriptors,
                                 matches, descriptorType, matcherType, selectorType, 
                                 algoCharacteristic.matcherElapsedTime.at(imgIndex));

                algoCharacteristic.numMatchedKpts.at(imgIndex) = matches.size();

                // store matches in current data frame
                (dataBuffer.end() - 1)->kptMatches = matches;

                cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                
                /* TRACK 3D OBJECT BOUNDING BOXES */

                //// STUDENT ASSIGNMENT
                //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous 
                ////frame (implement ->matchBoundingBoxes)
                map<int, int> bbBestMatches;
                // associate bounding boxes between current and previous frame using keypoint matches
                matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); 
                //// EOF STUDENT ASSIGNMENT

                // store matches in current data frame
                (dataBuffer.end()-1)->bbMatches = bbBestMatches;

                // visualize matches between current and previous image
                bVis = false;
                if (bVis)
                {
                    cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                    cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                    (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                    matches, matchImg,
                                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                                    vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                    string windowName = "Matching keypoints between two camera images";
                    cv::namedWindow(windowName, 7);
                    cv::imshow(windowName, matchImg);

                    //cv:imwrite("../output/" + detectorType + "_" + algoCharacteristic.descriptor + "_" + to_string(imgIndex) + ".jpg", matchImg);

                    cout << "Press key to continue to next image" << endl;
                    cv::waitKey(0); // wait for key to be pressed
                }
                bVis = false;

                cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


                /* COMPUTE TTC ON OBJECT IN FRONT */

                // loop over all BB match pairs
                for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin();
                     it1 != (dataBuffer.end() - 1)->bbMatches.end();
                     ++it1)
                {
                    // find bounding boxes associates with current match
                    BoundingBox *prevBB, *currBB;
                    for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin();
                         it2 != (dataBuffer.end() - 1)->boundingBoxes.end();
                         ++it2)
                    {
                        if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                        {
                            currBB = &(*it2);
                        }
                    }

                    for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin();
                         it2 != (dataBuffer.end() - 2)->boundingBoxes.end();
                         ++it2)
                    {
                        if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                        {
                            prevBB = &(*it2);
                        }
                    }

                    // compute TTC for current match

                    // only compute TTC if we have Lidar points
                    if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) 
                    {
                        //// STUDENT ASSIGNMENT
                        //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                        double ttcLidar; 
                        computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                        algoCharacteristic.lidarTTC.at(imgIndex) = ttcLidar;
                        //// EOF STUDENT ASSIGNMENT

                        //// STUDENT ASSIGNMENT
                        //// TASK FP.3 -> assign enclosed keypoint matches to bounding box 
                        ////              (implement -> clusterKptMatchesWithROI)
                        //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                        double ttcCamera;
                        clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints,
                                                 (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches); 

                        computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                         currBB->kptMatches, sensorFrameRate, ttcCamera);
                        
                        algoCharacteristic.cameraTTC.at(imgIndex) = ttcCamera;

                        algoCharacteristic.enoughLidarCameraPointsDetected.at(imgIndex) = true;


                        //// EOF STUDENT ASSIGNMENT

                        bVis = true;
                        if (bVis)
                        {
                            cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                            showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                            cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y),
                                          cv::Point(currBB->roi.x + currBB->roi.width,
                                          currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                            
                            char str[200];
                            sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                            putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                            string windowName = "Final Results : TTC";
                            cv::namedWindow(windowName, 4);
                            cv::imshow(windowName, visImg);
                            //cv::imwrite("../output_fusion/" + detectorType + "_" + algoCharacteristic.descriptor + "_" + to_string(imgIndex) + ".jpg", visImg);

                            cout << "Press key to continue to next frame" << endl;
                            cv::waitKey(0);
                        }
                        bVis = false;

                    } // eof TTC computation
                    else
                    {
                        algoCharacteristic.enoughLidarCameraPointsDetected.at(imgIndex) = false;
                    }
                    
                } // eof loop over all BB matches            

            }

        } // eof loop over all images
    }
    summarizeAndGenerateOutputReports(algoCharacteristics, numImages);
    return 0;
}

void InitAlgoCombinations(vector<AlgoCharacteristics>& algoChars)
 {
    vector<string> detectors = { "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE",  "SIFT" };
    vector<string> descriptors = { "BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT" };
    vector<string> matchers = { "MAT_BF" };
    vector<string> selectors = { "SEL_KNN" };

    for (auto detector : detectors) 
    {
        for (auto descriptor : descriptors)
        {
            for (auto matcher : matchers) 
            {
                for (auto selector : selectors) 
                {
                    if ((descriptor.compare("AKAZE") == 0) && (detector.compare("AKAZE") != 0) || (descriptor.compare("ORB") == 0) && (detector.compare("SIFT") == 0))
                    { 
                        continue; 
                    }

                    AlgoCharacteristics ac;
                    ac.detector = detector;
                    ac.descriptor = descriptor;
                    ac.matcher = matcher;
                    ac.selector = selector;

                    algoChars.push_back(ac);
                }
            }
        }
    }
}

void summarizeAndGenerateOutputReports(vector<AlgoCharacteristics> &algoOutputs, size_t numImages)
 {
    ofstream reportFile{"../report/Report.csv"};
    ofstream summaryFile{"../report/Summary.csv"};

    int totalKptsPerAllImages;
    int totalDescriptorsPerAllImages;
    int totalMatchedKptsPerAllImages;
    double totalDetectorDescritorTimeElapsed;
    double totalTimeElapsed;
    string algoCombo;

    reportFile << "IMAGE, DETECTOR, DESCRIPTOR, MATCHER, SELECTOR, #KEYPOINTS, #DESCRIPTORS, #MATCHED KEYPOINTS, TTC CAMERA, TTC LIDAR, DETECTOR ELAPSED TIME, DESCRIPTOR ELAPSED TIME, MATCHER ELAPSED TIME" << endl;
    summaryFile << "ALGO COMBO, #KEYPOINTS, #DESCRIPTORS, #MATCHED KEYPOINTS, DETECTOR & DESCRIPTOR TIME(ms), TOTAL TIME ELAPSED(ms)" << endl;

    for (auto &algoOutput : algoOutputs)
    {
        totalKptsPerAllImages = 0;
        totalDescriptorsPerAllImages = 0;
        totalMatchedKptsPerAllImages = 0;
        totalDetectorDescritorTimeElapsed = 0.0;
        totalTimeElapsed = 0.0;
        algoCombo = algoOutput.detector + "+" + algoOutput.descriptor;

        for(int imIndex = 1; imIndex < numImages; imIndex++)
        {
            reportFile << imIndex << ", " 
                    << algoOutput.detector << ", " 
                    << algoOutput.descriptor << ", " 
                    << algoOutput.matcher << ", " 
                    << algoOutput.selector << ", " 
                    << algoOutput.numKpts.at(imIndex)<< ", "
                    << algoOutput.numDescriptors.at(imIndex) << ", " 
                    << algoOutput.numMatchedKpts.at(imIndex) << ", " 
                    << algoOutput.cameraTTC.at(imIndex) << ", " 
                    << algoOutput.lidarTTC.at(imIndex) << ", " 
                    << algoOutput.detectorElapsedTime.at(imIndex) << ", " 
                    << algoOutput.descriptorElapsedTime.at(imIndex) << ", " 
                    << algoOutput.matcherElapsedTime.at(imIndex) << endl;

            totalKptsPerAllImages += algoOutput.numKpts.at(imIndex);
            totalDescriptorsPerAllImages += algoOutput.numDescriptors.at(imIndex);
            totalMatchedKptsPerAllImages += algoOutput.numMatchedKpts.at(imIndex);
            totalDetectorDescritorTimeElapsed += (algoOutput.detectorElapsedTime.at(imIndex) + 
                                                algoOutput.descriptorElapsedTime.at(imIndex)) ;
            totalTimeElapsed += totalDetectorDescritorTimeElapsed + algoOutput.matcherElapsedTime.at(imIndex);
        }
        summaryFile << algoCombo << ", " 
                    << totalKptsPerAllImages << ", " 
                    << totalDescriptorsPerAllImages << ", " 
                    << totalMatchedKptsPerAllImages << ", " 
                    << totalDetectorDescritorTimeElapsed << ", " 
                    << totalTimeElapsed << endl;
    }

    reportFile.close();
    summaryFile.close();

    std::cout << "Report Generated at " << "../report/Report.csv" << std::endl;
    std::cout << "Summary Generated at " << "../report/Summary.csv" << std::endl;
}
