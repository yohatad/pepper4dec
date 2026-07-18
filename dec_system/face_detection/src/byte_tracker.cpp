/* byte_tracker.cpp
 *
 * Implements the ByteTrack multi-object tracker declared in byte_tracker.h.
 * See that header for a summary of what's being ported.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 06, 2026
 * Version: v1.0 - C++ port of supervision.tracker.byte_tracker (Python)
 */

#include "face_detection/byte_tracker.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace byte_tracker {

// ── KalmanFilter ─────────────────────────────────────────────────────────────

KalmanFilter::KalmanFilter() {
    motion_mat_.setIdentity();
    for (int i = 0; i < 4; ++i) {
        motion_mat_(i, 4 + i) = 1.0;
    }
    update_mat_.setZero();
    update_mat_(0, 0) = 1.0;
    update_mat_(1, 1) = 1.0;
    update_mat_(2, 2) = 1.0;
    update_mat_(3, 3) = 1.0;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::initiate(const Eigen::Vector4d& measurement) const {
    Eigen::VectorXd mean(8);
    mean.head<4>() = measurement;
    mean.tail<4>().setZero();

    double h = measurement[3];
    Eigen::VectorXd std_dev(8);
    std_dev << 2 * std_weight_position_ * h, 2 * std_weight_position_ * h, 1e-2, 2 * std_weight_position_ * h,
        10 * std_weight_velocity_ * h, 10 * std_weight_velocity_ * h, 1e-5, 10 * std_weight_velocity_ * h;

    Eigen::MatrixXd covariance = std_dev.array().square().matrix().asDiagonal();
    return {mean, covariance};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::predict(
    const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance) const {
    double h = mean[3];
    Eigen::VectorXd std_dev(8);
    std_dev << std_weight_position_ * h, std_weight_position_ * h, 1e-2, std_weight_position_ * h,
        std_weight_velocity_ * h, std_weight_velocity_ * h, 1e-5, std_weight_velocity_ * h;
    Eigen::MatrixXd motion_cov = std_dev.array().square().matrix().asDiagonal();

    Eigen::VectorXd new_mean = motion_mat_ * mean;
    Eigen::MatrixXd new_cov = motion_mat_ * covariance * motion_mat_.transpose() + motion_cov;
    return {new_mean, new_cov};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::project(
    const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance) const {
    double h = mean[3];
    Eigen::Vector4d std_dev(std_weight_position_ * h, std_weight_position_ * h, 1e-1, std_weight_position_ * h);
    Eigen::Matrix4d innovation_cov = std_dev.array().square().matrix().asDiagonal();

    Eigen::VectorXd new_mean = update_mat_ * mean;
    Eigen::MatrixXd new_cov = update_mat_ * covariance * update_mat_.transpose() + innovation_cov;
    return {new_mean, new_cov};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::update(
    const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance, const Eigen::Vector4d& measurement) const {
    auto [projected_mean, projected_cov] = project(mean, covariance);

    // Kalman gain K = (covariance * H^T) * projected_cov^-1, solved via Cholesky
    // rather than an explicit inverse (mirrors scipy.linalg.cho_factor/cho_solve).
    Eigen::MatrixXd b = (covariance * update_mat_.transpose()).transpose();
    Eigen::LLT<Eigen::MatrixXd> chol(projected_cov);
    Eigen::MatrixXd kalman_gain = chol.solve(b).transpose();

    Eigen::VectorXd innovation = measurement - projected_mean;
    Eigen::VectorXd new_mean = mean + kalman_gain * innovation;
    Eigen::MatrixXd new_covariance = covariance - kalman_gain * projected_cov * kalman_gain.transpose();
    return {new_mean, new_covariance};
}

// ── STrack ───────────────────────────────────────────────────────────────────

STrack::STrack(const Eigen::Vector4d& tlwh, float score_in, int minimum_consecutive_frames_in)
    : score(score_in),
      minimum_consecutive_frames(minimum_consecutive_frames_in),
      raw_tlwh_(tlwh) {}

void STrack::predict(const KalmanFilter& kf) {
    Eigen::VectorXd mean_state = mean;
    if (state != TrackState::Tracked) {
        mean_state[7] = 0.0;
    }
    auto [new_mean, new_cov] = kf.predict(mean_state, covariance);
    mean = new_mean;
    covariance = new_cov;
}

void STrack::activate(const KalmanFilter& kf, int frame_id_in, int internal_id, int* external_id_counter) {
    kalman_filter_ = &kf;
    internal_track_id = internal_id;
    std::tie(mean, covariance) = kf.initiate(tlwhToXyah(raw_tlwh_));

    tracklet_len = 0;
    state = TrackState::Tracked;
    if (frame_id_in == 1) {
        is_activated = true;
    }

    if (minimum_consecutive_frames == 1) {
        external_track_id = (*external_id_counter)++;
    }

    frame_id = frame_id_in;
    start_frame = frame_id_in;
    has_state = true;
}

void STrack::reActivate(const STrack& new_track, int frame_id_in) {
    std::tie(mean, covariance) = kalman_filter_->update(mean, covariance, tlwhToXyah(new_track.tlwh()));
    tracklet_len = 0;
    state = TrackState::Tracked;
    frame_id = frame_id_in;
    score = new_track.score;
}

void STrack::update(const STrack& new_track, int frame_id_in, int* external_id_counter) {
    frame_id = frame_id_in;
    tracklet_len++;

    std::tie(mean, covariance) = kalman_filter_->update(mean, covariance, tlwhToXyah(new_track.tlwh()));
    state = TrackState::Tracked;
    if (tracklet_len == minimum_consecutive_frames) {
        is_activated = true;
        if (external_track_id == -1) {
            external_track_id = (*external_id_counter)++;
        }
    }
    score = new_track.score;
}

Eigen::Vector4d STrack::tlwh() const {
    if (!has_state) {
        return raw_tlwh_;
    }
    Eigen::Vector4d ret = mean.head<4>();
    ret[2] *= ret[3];
    ret[0] -= ret[2] / 2.0;
    ret[1] -= ret[3] / 2.0;
    return ret;
}

Eigen::Vector4d STrack::tlbr() const {
    return tlwhToTlbr(tlwh());
}

Eigen::Vector4d STrack::tlwhToXyah(const Eigen::Vector4d& tlwh_in) {
    Eigen::Vector4d ret = tlwh_in;
    ret[0] += ret[2] / 2.0;
    ret[1] += ret[3] / 2.0;
    ret[2] /= ret[3];
    return ret;
}

Eigen::Vector4d STrack::tlbrToTlwh(const Eigen::Vector4d& tlbr_in) {
    Eigen::Vector4d ret = tlbr_in;
    ret[2] -= ret[0];
    ret[3] -= ret[1];
    return ret;
}

Eigen::Vector4d STrack::tlwhToTlbr(const Eigen::Vector4d& tlwh_in) {
    Eigen::Vector4d ret = tlwh_in;
    ret[2] += ret[0];
    ret[3] += ret[1];
    return ret;
}

// ── Matching helpers ─────────────────────────────────────────────────────────

namespace matching {

Eigen::MatrixXd boxIouBatch(const std::vector<Eigen::Vector4d>& boxes_true,
                            const std::vector<Eigen::Vector4d>& boxes_detection) {
    int n = static_cast<int>(boxes_true.size());
    int m = static_cast<int>(boxes_detection.size());
    Eigen::MatrixXd iou = Eigen::MatrixXd::Zero(n, m);
    if (n == 0 || m == 0) return iou;

    for (int i = 0; i < n; ++i) {
        double ax1 = boxes_true[i][0], ay1 = boxes_true[i][1];
        double ax2 = boxes_true[i][2], ay2 = boxes_true[i][3];
        double area_a = std::max(0.0, ax2 - ax1) * std::max(0.0, ay2 - ay1);
        for (int j = 0; j < m; ++j) {
            double bx1 = boxes_detection[j][0], by1 = boxes_detection[j][1];
            double bx2 = boxes_detection[j][2], by2 = boxes_detection[j][3];
            double area_b = std::max(0.0, bx2 - bx1) * std::max(0.0, by2 - by1);

            double ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
            double ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
            double iw = std::max(0.0, ix2 - ix1), ih = std::max(0.0, iy2 - iy1);
            double inter = iw * ih;

            double denom = area_a + area_b - inter;
            iou(i, j) = denom > 0.0 ? inter / denom : 0.0;
        }
    }
    return iou;
}

std::vector<std::pair<int, int>> hungarianAssignment(const Eigen::MatrixXd& cost) {
    int rows = static_cast<int>(cost.rows());
    int cols = static_cast<int>(cost.cols());
    if (rows == 0 || cols == 0) return {};

    bool transposed = rows > cols;
    int n = transposed ? cols : rows;
    int m = transposed ? rows : cols;
    auto at = [&](int i, int j) -> double { return transposed ? cost(j, i) : cost(i, j); };

    const double INF = std::numeric_limits<double>::max() / 2.0;
    std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(m + 1, INF);
        std::vector<bool> used(m + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0];
            int j1 = -1;
            double delta = INF;
            for (int j = 1; j <= m; ++j) {
                if (!used[j]) {
                    double cur = at(i0 - 1, j - 1) - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    std::vector<std::pair<int, int>> result;
    for (int j = 1; j <= m; ++j) {
        if (p[j] > 0) {
            int i = p[j] - 1;
            int jj = j - 1;
            if (transposed) {
                result.push_back({jj, i});
            } else {
                result.push_back({i, jj});
            }
        }
    }
    return result;
}

AssignmentResult linearAssignment(Eigen::MatrixXd cost, double thresh) {
    AssignmentResult result;
    int rows = static_cast<int>(cost.rows());
    int cols = static_cast<int>(cost.cols());

    if (rows == 0 || cols == 0) {
        for (int i = 0; i < rows; ++i) result.unmatched_rows.push_back(i);
        for (int j = 0; j < cols; ++j) result.unmatched_cols.push_back(j);
        return result;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (cost(i, j) > thresh) cost(i, j) = thresh + 1e-4;
        }
    }

    auto raw_matches = hungarianAssignment(cost);
    std::vector<bool> row_matched(rows, false), col_matched(cols, false);
    for (const auto& [r, c] : raw_matches) {
        if (cost(r, c) <= thresh) {
            result.matches.push_back({r, c});
            row_matched[r] = true;
            col_matched[c] = true;
        }
    }
    for (int i = 0; i < rows; ++i) {
        if (!row_matched[i]) result.unmatched_rows.push_back(i);
    }
    for (int j = 0; j < cols; ++j) {
        if (!col_matched[j]) result.unmatched_cols.push_back(j);
    }
    return result;
}

namespace {
std::vector<Eigen::Vector4d> tlbrsOf(const std::vector<STrack>& tracks) {
    std::vector<Eigen::Vector4d> out;
    out.reserve(tracks.size());
    for (const auto& t : tracks) out.push_back(t.tlbr());
    return out;
}
std::vector<Eigen::Vector4d> tlbrsOf(const std::vector<STrack*>& tracks) {
    std::vector<Eigen::Vector4d> out;
    out.reserve(tracks.size());
    for (const auto* t : tracks) out.push_back(t->tlbr());
    return out;
}
}  // namespace

Eigen::MatrixXd iouDistance(const std::vector<STrack>& a, const std::vector<STrack>& b) {
    return 1.0 - boxIouBatch(tlbrsOf(a), tlbrsOf(b)).array();
}

Eigen::MatrixXd iouDistance(const std::vector<STrack*>& a, const std::vector<STrack>& b) {
    return 1.0 - boxIouBatch(tlbrsOf(a), tlbrsOf(b)).array();
}

Eigen::MatrixXd fuseScore(Eigen::MatrixXd cost_matrix, const std::vector<STrack>& detections) {
    int rows = static_cast<int>(cost_matrix.rows());
    int cols = static_cast<int>(cost_matrix.cols());
    if (rows == 0 || cols == 0) return cost_matrix;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double iou_sim = 1.0 - cost_matrix(i, j);
            double fuse_sim = iou_sim * detections[j].score;
            cost_matrix(i, j) = 1.0 - fuse_sim;
        }
    }
    return cost_matrix;
}

}  // namespace matching

// ── ByteTrack ────────────────────────────────────────────────────────────────

namespace {
// Joins two track lists, de-duplicating by internal_track_id (first occurrence wins).
std::vector<STrack> joinTracks(const std::vector<STrack>& a, const std::vector<STrack>& b) {
    std::unordered_set<int> seen;
    std::vector<STrack> result;
    result.reserve(a.size() + b.size());
    for (const auto& list : {a, b}) {
        for (const auto& t : list) {
            if (seen.insert(t.internal_track_id).second) {
                result.push_back(t);
            }
        }
    }
    return result;
}

// Returns tracks in `a` whose internal_track_id doesn't appear in `b`.
std::vector<STrack> subTracks(const std::vector<STrack>& a, const std::vector<STrack>& b) {
    std::unordered_set<int> ids_b;
    for (const auto& t : b) ids_b.insert(t.internal_track_id);
    std::vector<STrack> result;
    for (const auto& t : a) {
        if (ids_b.find(t.internal_track_id) == ids_b.end()) result.push_back(t);
    }
    return result;
}

std::pair<std::vector<STrack>, std::vector<STrack>> removeDuplicateTracks(
    const std::vector<STrack>& tracks_a, const std::vector<STrack>& tracks_b) {
    Eigen::MatrixXd dist = matching::iouDistance(tracks_a, tracks_b);
    std::vector<bool> dup_a(tracks_a.size(), false), dup_b(tracks_b.size(), false);
    for (int i = 0; i < dist.rows(); ++i) {
        for (int j = 0; j < dist.cols(); ++j) {
            if (dist(i, j) < 0.15) {
                int time_a = tracks_a[i].frame_id - tracks_a[i].start_frame;
                int time_b = tracks_b[j].frame_id - tracks_b[j].start_frame;
                if (time_a > time_b) {
                    dup_b[j] = true;
                } else {
                    dup_a[i] = true;
                }
            }
        }
    }
    std::vector<STrack> result_a, result_b;
    for (size_t i = 0; i < tracks_a.size(); ++i) {
        if (!dup_a[i]) result_a.push_back(tracks_a[i]);
    }
    for (size_t j = 0; j < tracks_b.size(); ++j) {
        if (!dup_b[j]) result_b.push_back(tracks_b[j]);
    }
    return {result_a, result_b};
}
}  // namespace

ByteTrack::ByteTrack(float track_activation_threshold, int lost_track_buffer, float minimum_matching_threshold,
                      int frame_rate, int minimum_consecutive_frames)
    : track_activation_threshold_(track_activation_threshold),
      minimum_matching_threshold_(minimum_matching_threshold),
      det_thresh_(track_activation_threshold + 0.1f),
      max_time_lost_(static_cast<int>(frame_rate / 30.0 * lost_track_buffer)),
      minimum_consecutive_frames_(minimum_consecutive_frames) {}

void ByteTrack::reset() {
    frame_id_ = 0;
    internal_id_counter_ = 0;
    external_id_counter_ = 1;
    tracked_tracks_.clear();
    lost_tracks_.clear();
    removed_tracks_.clear();
}

std::vector<STrack> ByteTrack::update(const std::vector<Eigen::Vector4d>& boxes, const std::vector<float>& scores) {
    frame_id_++;
    std::vector<STrack> activated_stracks;
    std::vector<STrack> refind_stracks;
    std::vector<STrack> lost_stracks;
    std::vector<STrack> removed_stracks;

    std::vector<Eigen::Vector4d> dets, dets_second;
    std::vector<float> scores_keep, scores_second;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (scores[i] > track_activation_threshold_) {
            dets.push_back(boxes[i]);
            scores_keep.push_back(scores[i]);
        } else if (scores[i] > 0.1f && scores[i] < track_activation_threshold_) {
            dets_second.push_back(boxes[i]);
            scores_second.push_back(scores[i]);
        }
    }

    std::vector<STrack> detections;
    for (size_t i = 0; i < dets.size(); ++i) {
        detections.emplace_back(STrack::tlbrToTlwh(dets[i]), scores_keep[i], minimum_consecutive_frames_);
    }

    // Split currently-tracked tracks into unconfirmed (never fully activated)
    // and confirmed (is_activated) tracks.
    std::vector<STrack*> unconfirmed;
    std::vector<STrack*> tracked_stracks;
    for (auto& track : tracked_tracks_) {
        if (!track.is_activated) {
            unconfirmed.push_back(&track);
        } else {
            tracked_stracks.push_back(&track);
        }
    }

    // Step 2: first association, with high-score detection boxes.
    std::vector<STrack*> strack_pool = tracked_stracks;
    for (auto& track : lost_tracks_) strack_pool.push_back(&track);

    for (auto* track : strack_pool) track->predict(kalman_filter_);

    Eigen::MatrixXd dists = matching::iouDistance(strack_pool, detections);
    dists = matching::fuseScore(dists, detections);
    auto assoc1 = matching::linearAssignment(dists, minimum_matching_threshold_);

    std::vector<bool> detection_used(detections.size(), false);
    for (const auto& [itracked, idet] : assoc1.matches) {
        STrack* track = strack_pool[itracked];
        const STrack& det = detections[idet];
        detection_used[idet] = true;
        if (track->state == TrackState::Tracked) {
            track->update(det, frame_id_, &external_id_counter_);
            activated_stracks.push_back(*track);
        } else {
            track->reActivate(det, frame_id_);
            refind_stracks.push_back(*track);
        }
    }

    // Step 3: second association, with low-score detection boxes.
    std::vector<STrack> detections_second;
    for (size_t i = 0; i < dets_second.size(); ++i) {
        detections_second.emplace_back(
            STrack::tlbrToTlwh(dets_second[i]), scores_second[i], minimum_consecutive_frames_);
    }

    std::vector<STrack*> r_tracked_stracks;
    for (int it : assoc1.unmatched_rows) {
        if (strack_pool[it]->state == TrackState::Tracked) r_tracked_stracks.push_back(strack_pool[it]);
    }

    Eigen::MatrixXd dists2 = matching::iouDistance(r_tracked_stracks, detections_second);
    auto assoc2 = matching::linearAssignment(dists2, 0.5);

    std::vector<bool> second_matched(r_tracked_stracks.size(), false);
    for (const auto& [itracked, idet] : assoc2.matches) {
        STrack* track = r_tracked_stracks[itracked];
        const STrack& det = detections_second[idet];
        second_matched[itracked] = true;
        if (track->state == TrackState::Tracked) {
            track->update(det, frame_id_, &external_id_counter_);
            activated_stracks.push_back(*track);
        } else {
            track->reActivate(det, frame_id_);
            refind_stracks.push_back(*track);
        }
    }
    for (size_t it = 0; it < r_tracked_stracks.size(); ++it) {
        if (!second_matched[it]) {
            STrack* track = r_tracked_stracks[it];
            if (track->state != TrackState::Lost) {
                track->state = TrackState::Lost;
                lost_stracks.push_back(*track);
            }
        }
    }

    // Unconfirmed tracks: usually tracks with only one beginning frame.
    std::vector<STrack> remaining_detections;
    std::vector<int> remaining_original_idx;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!detection_used[i]) {
            remaining_detections.push_back(detections[i]);
            remaining_original_idx.push_back(static_cast<int>(i));
        }
    }

    Eigen::MatrixXd dists3 = matching::iouDistance(unconfirmed, remaining_detections);
    dists3 = matching::fuseScore(dists3, remaining_detections);
    auto assoc3 = matching::linearAssignment(dists3, 0.7);

    std::vector<bool> unconfirmed_matched(unconfirmed.size(), false);
    std::vector<bool> remaining_used(remaining_detections.size(), false);
    for (const auto& [itracked, idet] : assoc3.matches) {
        unconfirmed[itracked]->update(remaining_detections[idet], frame_id_, &external_id_counter_);
        activated_stracks.push_back(*unconfirmed[itracked]);
        unconfirmed_matched[itracked] = true;
        remaining_used[idet] = true;
    }
    for (size_t it = 0; it < unconfirmed.size(); ++it) {
        if (!unconfirmed_matched[it]) {
            unconfirmed[it]->state = TrackState::Removed;
            removed_stracks.push_back(*unconfirmed[it]);
        }
    }

    // Step 4: init new tracks from detections nobody claimed.
    for (size_t i = 0; i < remaining_detections.size(); ++i) {
        if (remaining_used[i]) continue;
        STrack& track = remaining_detections[i];
        if (track.score < det_thresh_) continue;
        track.activate(kalman_filter_, frame_id_, internal_id_counter_++, &external_id_counter_);
        activated_stracks.push_back(track);
    }

    // Step 5: age out lost tracks past the buffer.
    for (auto& track : lost_tracks_) {
        if (frame_id_ - track.frame_id > max_time_lost_) {
            track.state = TrackState::Removed;
            removed_stracks.push_back(track);
        }
    }

    std::vector<STrack> still_tracked;
    for (auto& track : tracked_tracks_) {
        if (track.state == TrackState::Tracked) still_tracked.push_back(track);
    }
    tracked_tracks_ = joinTracks(still_tracked, activated_stracks);
    tracked_tracks_ = joinTracks(tracked_tracks_, refind_stracks);
    lost_tracks_ = subTracks(lost_tracks_, tracked_tracks_);
    for (auto& t : lost_stracks) lost_tracks_.push_back(t);
    lost_tracks_ = subTracks(lost_tracks_, removed_tracks_);
    removed_tracks_ = removed_stracks;
    std::tie(tracked_tracks_, lost_tracks_) = removeDuplicateTracks(tracked_tracks_, lost_tracks_);

    std::vector<STrack> output;
    for (const auto& track : tracked_tracks_) {
        if (track.is_activated) output.push_back(track);
    }
    return output;
}

Detections ByteTrack::updateWithDetections(const Detections& detections) {
    std::vector<Eigen::Vector4d> boxes = detections.xyxy;
    std::vector<float> scores = detections.confidence;

    std::vector<STrack> tracks = update(boxes, scores);

    Detections result;
    if (!tracks.empty()) {
        std::vector<Eigen::Vector4d> track_boxes;
        for (const auto& t : tracks) track_boxes.push_back(t.tlbr());

        Eigen::MatrixXd iou = matching::boxIouBatch(boxes, track_boxes);
        Eigen::MatrixXd iou_costs = 1.0 - iou.array();
        auto assoc = matching::linearAssignment(iou_costs, 0.5);

        for (const auto& [i_detection, i_track] : assoc.matches) {
            result.xyxy.push_back(detections.xyxy[i_detection]);
            result.confidence.push_back(detections.confidence[i_detection]);
            result.class_id.push_back(detections.class_id[i_detection]);
            result.tracker_id.push_back(tracks[i_track].external_track_id);
        }
    }
    return result;
}

}  // namespace byte_tracker
