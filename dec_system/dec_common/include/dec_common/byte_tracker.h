/* byte_tracker.h
 *
 * C++ port of the ByteTrack multi-object tracker from the `supervision`
 * Python package (supervision.tracker.byte_tracker). Shared by the face_detection
 * and person_detection nodes to assign persistent track IDs to per-frame
 * detections. Reproduces the
 * two-stage (high/low confidence) IoU association, the 8-state
 * constant-velocity Kalman filter, and the Hungarian (Kuhn-Munkres)
 * assignment used for both detection-to-track matching and duplicate-track
 * removal.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 06, 2026
 * Version: v1.0 - C++ port of supervision.tracker.byte_tracker (Python)
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#ifndef DEC_COMMON_BYTE_TRACKER_H
#define DEC_COMMON_BYTE_TRACKER_H

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

namespace byte_tracker {

// ── Detections ──────────────────────────────────────────────────────────────
// Mirrors the subset of supervision.Detections fields actually used here.
struct Detections {
    std::vector<Eigen::Vector4d> xyxy;   // (x1, y1, x2, y2) per detection
    std::vector<float> confidence;
    std::vector<int> class_id;
    std::vector<int> tracker_id;         // filled in by ByteTrack::updateWithDetections
};

// ── KalmanFilter ─────────────────────────────────────────────────────────────
// 8-state (x, y, aspect, height, vx, vy, va, vh) constant-velocity Kalman
// filter for bounding boxes in image space, identical in form to
// supervision's KalmanFilter (itself a bytetrack-style filter).
class KalmanFilter {
public:
    KalmanFilter();

    // measurement: (x, y, aspect, height). Returns (mean[8], covariance[8x8]).
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> initiate(const Eigen::Vector4d& measurement) const;
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict(
        const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance) const;
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> project(
        const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance) const;
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> update(
        const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance,
        const Eigen::Vector4d& measurement) const;

private:
    Eigen::Matrix<double, 8, 8> motion_mat_;
    Eigen::Matrix<double, 4, 8> update_mat_;
    double std_weight_position_ = 1.0 / 20.0;
    double std_weight_velocity_ = 1.0 / 160.0;
};

// ── STrack ───────────────────────────────────────────────────────────────────
enum class TrackState { New, Tracked, Lost, Removed };

class STrack {
public:
    STrack(const Eigen::Vector4d& tlwh, float score, int minimum_consecutive_frames);

    void predict(const KalmanFilter& kf);
    void activate(const KalmanFilter& kf, int frame_id, int internal_id, int* external_id_counter);
    void reActivate(const STrack& new_track, int frame_id);
    void update(const STrack& new_track, int frame_id, int* external_id_counter);

    Eigen::Vector4d tlwh() const;
    Eigen::Vector4d tlbr() const;
    static Eigen::Vector4d tlwhToXyah(const Eigen::Vector4d& tlwh);
    static Eigen::Vector4d tlbrToTlwh(const Eigen::Vector4d& tlbr);
    static Eigen::Vector4d tlwhToTlbr(const Eigen::Vector4d& tlwh);

    TrackState state = TrackState::New;
    bool is_activated = false;
    int start_frame = 0;
    int frame_id = 0;
    float score = 0.0f;
    int tracklet_len = 0;
    int minimum_consecutive_frames = 1;

    // NO_ID (-1) until assigned.
    int internal_track_id = -1;
    int external_track_id = -1;

    bool has_state = false;   // false until activate() runs (mirrors Python's mean is None)
    Eigen::VectorXd mean;     // 8-dim
    Eigen::MatrixXd covariance;  // 8x8

private:
    Eigen::Vector4d raw_tlwh_;  // raw tlwh, used only before activation
    const KalmanFilter* kalman_filter_ = nullptr;
};

// ── Matching helpers ─────────────────────────────────────────────────────────
namespace matching {

// Pairwise IoU between two batches of xyxy boxes. Returns rows.size() x cols.size().
Eigen::MatrixXd boxIouBatch(const std::vector<Eigen::Vector4d>& boxes_true,
                            const std::vector<Eigen::Vector4d>& boxes_detection);

// Solves the rectangular linear sum assignment problem (Hungarian / Kuhn-Munkres),
// minimizing total cost. Equivalent to scipy.optimize.linear_sum_assignment.
// Returns matched (row, col) index pairs; every row and every column appears
// at most once, with min(rows, cols) pairs returned.
std::vector<std::pair<int, int>> hungarianAssignment(const Eigen::MatrixXd& cost);

// Thresholded Hungarian assignment matching supervision's matching.linear_assignment:
// entries with cost > thresh are excluded from the returned matches (but still
// considered by the solver, matching scipy's clip-then-solve behavior).
struct AssignmentResult {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_rows;
    std::vector<int> unmatched_cols;
};
AssignmentResult linearAssignment(Eigen::MatrixXd cost, double thresh);

// 1 - IoU, for use as a cost matrix.
Eigen::MatrixXd iouDistance(const std::vector<STrack>& a, const std::vector<STrack>& b);
Eigen::MatrixXd iouDistance(const std::vector<STrack*>& a, const std::vector<STrack>& b);

// Fuses detection confidence into an IoU cost matrix (in place), matching
// supervision's matching.fuse_score.
Eigen::MatrixXd fuseScore(Eigen::MatrixXd cost_matrix, const std::vector<STrack>& detections);

}  // namespace matching

// ── ByteTrack ────────────────────────────────────────────────────────────────
class ByteTrack {
public:
    ByteTrack(float track_activation_threshold = 0.25f, int lost_track_buffer = 30,
              float minimum_matching_threshold = 0.8f, int frame_rate = 30,
              int minimum_consecutive_frames = 1);

    // Updates the tracker with the given detections and returns them with
    // tracker_id filled in; unmatched detections are dropped (tracker_id
    // would be -1), matching supervision's update_with_detections.
    Detections updateWithDetections(const Detections& detections);

    void reset();

private:
    std::vector<STrack> update(const std::vector<Eigen::Vector4d>& boxes, const std::vector<float>& scores);

    float track_activation_threshold_;
    float minimum_matching_threshold_;
    int frame_id_ = 0;
    float det_thresh_;
    int max_time_lost_;
    int minimum_consecutive_frames_;

    KalmanFilter kalman_filter_;

    std::vector<STrack> tracked_tracks_;
    std::vector<STrack> lost_tracks_;
    std::vector<STrack> removed_tracks_;

    int internal_id_counter_ = 0;
    int external_id_counter_ = 1;
};

}  // namespace byte_tracker

#endif  // DEC_COMMON_BYTE_TRACKER_H
