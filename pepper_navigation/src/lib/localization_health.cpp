/* localization_health.cpp
 *
 * See include/pepper_navigation/localization_health.h.
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

#include "pepper_navigation/localization_health.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>

namespace pepper_navigation
{

namespace
{

std::string seconds(double s)
{
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.0fs", s);
  return buf;
}

}  // namespace

LocalizationHealth::LocalizationHealth(const HealthParams & params)
: params_(params)
{
  for (const auto & name : params_.status_names) {
    entries_.push_back({name, std::nullopt, std::nullopt});
  }
}

void LocalizationHealth::report(const std::string & name, std::uint8_t level, double now)
{
  for (auto & e : entries_) {
    if (e.name != name) {
      continue;
    }
    e.at = now;
    const bool bad = level == kLevelError || (params_.warn_is_lost && level == kLevelWarn);
    if (!bad) {
      e.bad_since.reset();
    } else if (!e.bad_since) {
      // Also covers a first sighting that is already bad: the clock starts
      // now rather than treating "never healthy" as lost.
      e.bad_since = now;
    }
  }
}

bool LocalizationHealth::seen(const std::string & name) const
{
  return std::any_of(
    entries_.begin(), entries_.end(),
    [&name](const Entry & e) {return e.name == name && e.at;});
}

std::pair<bool, std::string> LocalizationHealth::worst(double now) const
{
  bool lost = false;
  std::string reason;
  for (const auto & e : entries_) {
    if (!e.at) {
      continue;  // never reported; not our business to guess
    }
    const double silent = now - *e.at;
    if (params_.stale_timeout > 0.0 && silent > params_.stale_timeout) {
      return {true, "no \"" + e.name + "\" status for " + seconds(silent)};
    }
    const double bad_for = e.bad_since ? now - *e.bad_since : 0.0;
    if (bad_for >= params_.lost_duration) {
      lost = true;
      reason = "\"" + e.name + "\" unhealthy for " + seconds(bad_for);
    }
  }
  return {lost, reason};
}

HealthStep LocalizationHealth::step(double now)
{
  HealthStep out;
  const bool any_seen = std::any_of(
    entries_.begin(), entries_.end(), [](const Entry & e) {return e.at.has_value();});
  if (!any_seen) {
    return out;  // nothing has reported yet; not our business to guess
  }

  const auto [lost, reason] = worst(now);
  if (!lost) {
    if (acting_) {
      out.recovered = true;
      acting_ = false;
    }
    return out;
  }

  out.reason = reason;
  if (!acting_) {
    out.became_lost = true;
    acting_ = true;
    last_cancel_.reset();
    out.request_recovery = params_.call_recovery;
  }

  // No cancel yet in this episode means one is due NOW. Treating "never" as
  // "just now" once kept the period gate permanently closed: LOST was
  // detected and nothing was ever cancelled.
  const bool due = !last_cancel_ || now - *last_cancel_ >= params_.recancel_period;
  if (params_.cancel_goals && due) {
    out.cancel_goals = true;
    last_cancel_ = now;
  }
  return out;
}

}  // namespace pepper_navigation
