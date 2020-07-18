#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType, double &elapsedTime)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {   
            descSource.convertTo(descSource, CV_32F);
        }
        if (descRef.type() != CV_32F)
        {   
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    if (selectorType.compare("SEL_NN") == 0)
    {
        double t = (double)cv::getTickCount();

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        elapsedTime = 1000 * t / 1.0;
        cout << " (KNN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        vector<vector<cv::DMatch>> knnMatches;

        double t = (double)cv::getTickCount();

        matcher->knnMatch(descSource, descRef, knnMatches, 2); // finds the 2 best matches

        double minDescDistRatio { 0.8 };

        for (auto it = knnMatches.begin(); it != knnMatches.end(); ++it)
        {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        elapsedTime = 1000 * t / 1.0;
        cout << " (KNN) with n=" << knnMatches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
}

void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, double &elapsedTime)
{
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;
        int octaves = 3;
        float patternScale = 1.0f;

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();

    extractor->compute(img, keypoints, descriptors);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    elapsedTime = 1000 * t / 1.0;

    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &elapsedTime, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    elapsedTime = 1000 * t / 1.0;

    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        visualizeResults(img, keypoints, "Shi-Tomasi Corner Detector Results");
        // cv::Mat visImage = img.clone();
        // cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        // string windowName = "Shi-Tomasi Corner Detector Results";
        // cv::namedWindow(windowName, 6);
        // imshow(windowName, visImage);
        // cv::waitKey(0);
    }
}

void visualizeResults(cv::Mat &img, vector<cv::KeyPoint> &keypoints, string windowName)
{
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &elapsedTime, bool bVis)
{
    int blockSize { 2 };
    int apertureSize { 3 };
    int minResponse { 100 };
    double k { 0.04 };

    double t = (double)cv::getTickCount();
  
    cv::Mat dst, dstNorm, dstNormScaled;
  
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
  
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
  
    cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  
    //cv::convertScaleAbs(dstNorm, dstNormScaled);

    double maxOverlap = 0.0;
  
    for (size_t j = 0; j < dstNorm.rows; j++)
    {
        for (size_t i = 0; i < dstNorm.cols; i++)
        {
            int response = (int)dstNorm.at<float>(j, i);
          
            if (response > minResponse)
            {
                cv::KeyPoint kp;
              
                kp.pt = cv::Point2f(i, j);
                kp.size = 2 * apertureSize;
                kp.response = response;
                
                bool bOverlap = false;

                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(kp, *it);

                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (kp.response > (*it).response)
                        {
                            *it = kp;
                            break;
                        }
                    }
                }

                if (!bOverlap)
                {
                    keypoints.push_back(kp);
                }
            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    elapsedTime = 1000 * t / 1.0;
    cout << "Harris detection with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        visualizeResults(img, keypoints, "Harris Corner Detector Results");
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, double &elapsedTime, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector; 

    if(detectorType.compare("FAST") == 0)
    {
        int threshold = 30;
        bool bNMS = true;

        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; //TYPE_7_12, TYPE_9_16, TYPE_5_8

        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    }
    else if(detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
    }
    else if(detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
    }
    else if(detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
    }
    else if(detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
    }

    double t = (double)cv::getTickCount();

    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    elapsedTime = 1000 * t / 1.0;
    cout << detectorType << " detection with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        visualizeResults(img, keypoints, detectorType + " Corner Detection Results");
    }
}