/* overt_attention_bms.cpp
 *
 * Boolean Map Saliency (BMS) — the bottom-up saliency operator used by
 * SaliencyNode. Split into its own translation unit so it can be linked into
 * overt_attention_lib and unit-tested; overt_attention_saliency.cpp carries
 * the node and its main(), which a test binary cannot link against.
 *
 * Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: June 12, 2026
 * Version: v1.0
 */

#include "overt_attention/overt_attention_interface.h"

//=============================================================================
// BooleanMapSaliency
//=============================================================================

BooleanMapSaliency::BooleanMapSaliency(int n_thresholds) : n_thresholds_(n_thresholds) {
    // Mirrors np.linspace(0, 1, n_thresholds + 1, endpoint=False)[1:]
    thresholds_.reserve(n_thresholds_);
    for (int i = 1; i <= n_thresholds_; ++i) {
        thresholds_.push_back(static_cast<double>(i) / (n_thresholds_ + 1));
    }
}

cv::Mat BooleanMapSaliency::activateBooleanMap(const cv::Mat& bool_map) {
    cv::Mat activation = bool_map.clone();
    int h = activation.rows;
    int w = activation.cols;
    cv::Mat ffill_mask = cv::Mat::zeros(h + 2, w + 2, CV_8UC1);

    for (int y = 0; y < h; ++y) {
        if (activation.at<uchar>(y, 0)) {
            cv::floodFill(activation, ffill_mask, cv::Point(0, y), cv::Scalar(0));
        }
        if (activation.at<uchar>(y, w - 1)) {
            cv::floodFill(activation, ffill_mask, cv::Point(w - 1, y), cv::Scalar(0));
        }
    }
    for (int x = 0; x < w; ++x) {
        if (activation.at<uchar>(0, x)) {
            cv::floodFill(activation, ffill_mask, cv::Point(x, 0), cv::Scalar(0));
        }
        if (activation.at<uchar>(h - 1, x)) {
            cv::floodFill(activation, ffill_mask, cv::Point(x, h - 1), cv::Scalar(0));
        }
    }

    return activation;
}

cv::Mat BooleanMapSaliency::computeSaliency(const cv::Mat& frame_bgr) {
    cv::Mat lab;
    cv::cvtColor(frame_bgr, lab, cv::COLOR_BGR2Lab);
    lab.convertTo(lab, CV_32F);

    double lab_min, lab_max;
    cv::minMaxLoc(lab.reshape(1), &lab_min, &lab_max);
    double lab_range = lab_max - lab_min;
    if (lab_range < 1e-6) {
        return cv::Mat::zeros(frame_bgr.rows, frame_bgr.cols, CV_32F);
    }
    lab = (lab - lab_min) / lab_range;

    int h = lab.rows, w = lab.cols;
    cv::Mat saliency = cv::Mat::zeros(h, w, CV_32F);

    std::vector<cv::Mat> lab_ch;
    cv::split(lab, lab_ch);

    for (double thresh : thresholds_) {
        for (int c = 0; c < 3; ++c) {
            cv::Mat bool_map = (lab_ch[c] > thresh);
            cv::Mat activation = activateBooleanMap(bool_map);
            cv::Mat activation_f;
            activation.convertTo(activation_f, CV_32F, 1.0 / 255.0);
            saliency += activation_f;
        }
    }

    saliency /= (n_thresholds_ * 3);
    return saliency;
}
