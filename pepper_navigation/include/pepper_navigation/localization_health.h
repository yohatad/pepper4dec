/* localization_health.h
 *
 * localization_watchdog's decisions, kept free of ROS so they can be tested
 * with time passed in (test/test_localization_health.cpp): which watched
 * diagnostics status is unhealthy, for how long, and when to cancel the active
 * navigation goals.
 *
 * A status is bad at ERROR (and at WARN with warn_is_lost). Badness is timed
 * from the FIRST bad report, not the last good one, so a gap in reporting
 * never counts as unhealthy; silence is stale_timeout's job, and only once a
 * status has been seen, so slow startup is not mistaken for a crash. A status
 * never seen is ignored, so a stack that publishes only some of the watched
 * names is not reported as permanently stale.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: November 17, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace pepper_navigation
{

// diagnostic_msgs/DiagnosticStatus levels.
constexpr std::uint8_t kLevelOk = 0;
constexpr std::uint8_t kLevelWarn = 1;
constexpr std::uint8_t kLevelError = 2;

struct HealthParams
{
  std::vector<std::string> status_names;
  double lost_duration = 5.0;     // s of continuous badness before acting
  double stale_timeout = 10.0;    // s of silence after a first report; 0 disables
  bool warn_is_lost = false;
  bool cancel_goals = true;
  bool call_recovery = false;
  double recancel_period = 3.0;   // s between cancels while still lost
};

// What one periodic check asks the node to do.
struct HealthStep
{
  bool became_lost = false;       // log the error; once per LOST episode
  bool recovered = false;         // log the release
  bool request_recovery = false;  // call the recovery service
  bool cancel_goals = false;      // cancel every navigation goal now
  std::string reason;             // why it is lost, when it is
};

class LocalizationHealth
{
public:
  explicit LocalizationHealth(const HealthParams & params);

  // One status from /diagnostics. Names not watched are ignored.
  void report(const std::string & name, std::uint8_t level, double now);

  // The worst watched status: (lost, reason) over the ones seen so far.
  std::pair<bool, std::string> worst(double now) const;

  // The periodic check (check_period): updates the LOST state and says what
  // to do about it.
  HealthStep step(double now);

  // Currently treating localization as lost.
  bool acting() const {return acting_;}

  // Whether this watched status has been reported at least once.
  bool seen(const std::string & name) const;

private:
  struct Entry
  {
    std::string name;
    std::optional<double> bad_since;  // start of the current run of bad reports
    std::optional<double> at;         // last report of any level
  };

  HealthParams params_;
  std::vector<Entry> entries_;        // in status_names order
  bool acting_ = false;
  std::optional<double> last_cancel_;
};

}  // namespace pepper_navigation
