
#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>

struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
    
    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs
    
    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust

    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
};

struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int,int> bbMatches; // bounding box matches between previous and current frame
};

struct AlgoCharacteristics {
    std::string detector;
    std::string descriptor;
    std::string matcher;
    std::string selector;
    std::array<int, 30> numKpts;
    std::array<int, 30> numKptsVehicle;
    std::array<int, 30> numDescriptors;
    std::array<int, 30> numMatchedKpts;
    std::array<double, 30> lidarTTC;
    std::array<double, 30> cameraTTC;
    std::array<bool, 30> enoughLidarCameraPointsDetected;
    std::array<double, 30> detectorElapsedTime;
    std::array<double, 30> descriptorElapsedTime;
    std::array<double, 30> matcherElapsedTime;
};


#endif /* dataStructures_h */
