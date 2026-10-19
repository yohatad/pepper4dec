/* topic_liveness.h
 *
 * "Have all the topics I depend on been heard from recently?", kept free of
 * ROS types so it can be tested directly (test/test_topic_liveness.cpp).
 *
 * Why this exists: MEASURED on slam_20260823_aligned, Humble's
 * collision_monitor HOLDS ITS LAST VERDICT when a source goes stale. With the
 * laser as its only enabled source, feeding it a clear reading and then going
 * silent gave 161 consecutive full-speed passes with no sensor data at all --
 * source_timeout only decides whether a reading is used, not whether the
 * robot may keep moving without one. Nothing else in the chain catches it
 * either: the driver's cmd_vel_timeout fires only if cmd_vel stops, and it
 * does not, and robot_watchdog.py watches the ALMemory heartbeat rather than
 * sensor liveness.
 *
 * A topic never heard from counts as stale, not as fine: a source that failed
 * to connect at startup is the case this most needs to catch.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: October 19, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#ifndef PEPPER_NAVIGATION__TOPIC_LIVENESS_H_
#define PEPPER_NAVIGATION__TOPIC_LIVENESS_H_

#include <map>
#include <string>
#include <vector>

namespace pepper_navigation
{

/// Sentinel for "never heard from". Any real clock reading exceeds it.
constexpr double kNeverSeen = -1e30;

class TopicLiveness
{
public:
  TopicLiveness() = default;

  TopicLiveness(const std::vector<std::string> & topics, double timeout)
  : timeout_(timeout)
  {
    for (const auto & t : topics) {seen_[t] = kNeverSeen;}
  }

  /// Nothing required means the check is off, not that everything is stale.
  bool watching() const {return !seen_.empty();}

  void mark(const std::string & topic, double now_s)
  {
    auto it = seen_.find(topic);
    if (it != seen_.end()) {it->second = now_s;}
  }

  bool stale(const std::string & topic, double now_s) const
  {
    auto it = seen_.find(topic);
    if (it == seen_.end()) {return false;}
    if (it->second == kNeverSeen) {return true;}
    return (now_s - it->second) > timeout_;
  }

  /// Every required topic that is currently stale, in a stable order.
  std::vector<std::string> stale_topics(double now_s) const
  {
    std::vector<std::string> out;
    for (const auto & kv : seen_) {
      if (stale(kv.first, now_s)) {out.push_back(kv.first);}
    }
    return out;
  }

  bool any_stale(double now_s) const
  {
    for (const auto & kv : seen_) {
      if (stale(kv.first, now_s)) {return true;}
    }
    return false;
  }

  /// Topics that have never been heard from, so the node can keep trying to
  /// bind them instead of giving up on a publisher that started late.
  std::vector<std::string> unseen() const
  {
    std::vector<std::string> out;
    for (const auto & kv : seen_) {
      if (kv.second == kNeverSeen) {out.push_back(kv.first);}
    }
    return out;
  }

private:
  std::map<std::string, double> seen_;   // ordered, so reports are stable
  double timeout_ = 1.0;
};

}  // namespace pepper_navigation

#endif  // PEPPER_NAVIGATION__TOPIC_LIVENESS_H_
