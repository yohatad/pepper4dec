/* test_localization_health.cpp
 *
 * localization_watchdog's decisions on simulated time. Every branch is time
 * based (lost_duration, stale_timeout, recancel_period), so the test owns the
 * clock: it steps in 0.1 s, reports statuses every 1 s, and runs the periodic
 * check every 1 s, as the node does with check_period 1.0. Cancels are
 * recorded with the time they were asked for.
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

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "pepper_navigation/localization_health.h"

using pepper_navigation::HealthParams;
using pepper_navigation::kLevelError;
using pepper_navigation::kLevelOk;
using pepper_navigation::kLevelWarn;
using pepper_navigation::LocalizationHealth;

namespace
{

const char kLock[] = "fastlio_localization: pose lock";
const char kGate[] = "fastlio_localization: map -> odom gate";
constexpr double kStart = 100.0;
constexpr double kStep = 0.1;

bool contains(const std::string & text, const std::string & part)
{
  return text.find(part) != std::string::npos;
}

HealthParams params(bool both = false)
{
  HealthParams p;
  p.status_names = both ? std::vector<std::string>{kLock, kGate} :
  std::vector<std::string>{kLock};
  return p;  // lost 5 s, stale 10 s, recancel 3 s, cancel_goals on
}

// A watchdog on a simulated clock: statuses every 1 s, a check every 1 s.
class Sim
{
public:
  explicit Sim(const HealthParams & p)
  : health(p) {}

  // Advance `seconds`, reporting `levels` (none if empty) every `every` s.
  void run(double seconds, const std::map<std::string, std::uint8_t> & levels = {})
  {
    const int n = static_cast<int>(std::lround(seconds / kStep));
    for (int i = 0; i < n; ++i) {
      if (!levels.empty() && i % 10 == 0) {
        for (const auto & [name, level] : levels) {
          health.report(name, level, t);
        }
      }
      ++ticks_;
      t = kStart + ticks_ * kStep;
      if (ticks_ % 10 == 0) {
        const auto step = health.step(t);
        if (step.cancel_goals) {
          cancels.push_back(t);
        }
        if (step.became_lost) {
          became_lost.push_back(t);
        }
      }
    }
  }

  LocalizationHealth health;
  double t = kStart;
  std::vector<double> cancels;
  std::vector<double> became_lost;

private:
  std::int64_t ticks_ = 0;
};

}  // namespace

TEST(LocalizationHealth, NothingPublishedNeverCancels)
{
  Sim s(params());
  s.run(30.0);
  EXPECT_FALSE(s.health.acting());
  EXPECT_TRUE(s.cancels.empty());
}

TEST(LocalizationHealth, ErrorLongerThanLostDurationCancels)
{
  Sim s(params());
  s.run(2.0, {{kLock, kLevelOk}});
  const double first_error = s.t;
  s.run(7.0, {{kLock, kLevelError}});
  EXPECT_TRUE(s.health.acting());
  ASSERT_FALSE(s.cancels.empty()) << "no cancel after 7 s of ERROR";
  // Lost at first_error + lost_duration, acted on at the next 1 s check.
  EXPECT_GE(s.cancels[0], first_error + 5.0 - 1e-9);
  EXPECT_LE(s.cancels[0], first_error + 6.0 + kStep);
}

TEST(LocalizationHealth, ErrorShorterThanLostDurationDoesNotCancel)
{
  Sim s(params());
  s.run(2.0, {{kLock, kLevelOk}});
  s.run(3.5, {{kLock, kLevelError}});
  s.run(10.0, {{kLock, kLevelOk}});
  EXPECT_FALSE(s.health.acting());
  EXPECT_TRUE(s.cancels.empty());
}

TEST(LocalizationHealth, StatusNeverPublishedIsIgnored)
{
  Sim s(params(true));
  s.run(30.0, {{kLock, kLevelOk}});  // 3x stale_timeout without one gate status
  EXPECT_FALSE(s.health.acting());
  EXPECT_TRUE(s.cancels.empty());
}

TEST(LocalizationHealth, LockOkGateErrorIsLost)
{
  Sim s(params(true));
  s.run(2.0, {{kLock, kLevelOk}, {kGate, kLevelOk}});
  s.run(7.0, {{kLock, kLevelOk}, {kGate, kLevelError}});
  const auto [lost, reason] = s.health.worst(s.t);
  EXPECT_TRUE(lost);
  EXPECT_TRUE(s.health.acting());
  EXPECT_FALSE(s.cancels.empty());
  EXPECT_TRUE(contains(reason, kGate)) << reason;
}

TEST(LocalizationHealth, BothErrorNamesOne)
{
  Sim s(params(true));
  s.run(2.0, {{kLock, kLevelOk}, {kGate, kLevelOk}});
  s.run(7.0, {{kLock, kLevelError}, {kGate, kLevelError}});
  const auto [lost, reason] = s.health.worst(s.t);
  EXPECT_TRUE(lost);
  EXPECT_FALSE(s.cancels.empty());
  EXPECT_NE(contains(reason, kLock), contains(reason, kGate)) << reason;
}

TEST(LocalizationHealth, RecoveryClearsAndStopsCancelling)
{
  Sim s(params());
  s.run(2.0, {{kLock, kLevelOk}});
  s.run(7.0, {{kLock, kLevelError}});
  ASSERT_TRUE(s.health.acting());
  ASSERT_FALSE(s.cancels.empty());
  const auto n = s.cancels.size();
  s.run(1.0, {{kLock, kLevelOk}});
  EXPECT_FALSE(s.health.acting()) << "still holding navigation after a healthy status";
  s.run(15.0, {{kLock, kLevelOk}});
  EXPECT_EQ(s.cancels.size(), n) << "cancelled again after recovering";
}

TEST(LocalizationHealth, SilenceAfterFirstSightingIsLost)
{
  Sim s(params());
  s.run(2.0, {{kLock, kLevelOk}});
  s.run(8.5);  // silent past lost_duration, short of stale_timeout
  EXPECT_FALSE(s.health.acting()) << "silence after OK counted as unhealthy";
  EXPECT_TRUE(s.cancels.empty());
  s.run(2.5);  // now past stale_timeout (10 s since the last OK)
  const auto [lost, reason] = s.health.worst(s.t);
  EXPECT_TRUE(lost);
  EXPECT_TRUE(s.health.acting());
  EXPECT_FALSE(s.cancels.empty());
  EXPECT_EQ(reason.rfind(std::string("no \"") + kLock + "\" status for", 0), 0u) << reason;
}

TEST(LocalizationHealth, ReportingGapDoesNotCountAsUnhealthy)
{
  Sim s(params());
  s.run(2.0, {{kLock, kLevelOk}});
  s.run(4.0);  // a gap, still inside stale_timeout
  const double first_error = s.t;
  s.run(4.0, {{kLock, kLevelError}});
  EXPECT_TRUE(s.cancels.empty()) << "the gap before the ERROR was counted as unhealthy";
  s.run(3.0, {{kLock, kLevelError}});
  ASSERT_FALSE(s.cancels.empty());
  EXPECT_GE(s.cancels[0], first_error + 5.0 - 1e-9);
  const auto [lost, reason] = s.health.worst(s.t);
  EXPECT_EQ(reason.rfind(std::string("\"") + kLock + "\" unhealthy for", 0), 0u) << reason;
}

TEST(LocalizationHealth, ErrorThenSilenceStaysLost)
{
  Sim s(params());
  s.run(2.0, {{kLock, kLevelOk}});
  s.run(1.0, {{kLock, kLevelError}});  // the last word is ERROR, then nothing
  s.run(6.0);
  EXPECT_TRUE(s.health.acting());
  EXPECT_FALSE(s.cancels.empty());
}

TEST(LocalizationHealth, WarnIsNotLostByDefault)
{
  Sim s(params());
  s.run(2.0, {{kLock, kLevelOk}});
  s.run(8.0, {{kLock, kLevelWarn}});
  EXPECT_FALSE(s.health.acting());
  EXPECT_TRUE(s.cancels.empty());
}

TEST(LocalizationHealth, WarnIsLostWhenAsked)
{
  auto p = params();
  p.warn_is_lost = true;
  Sim s(p);
  s.run(2.0, {{kLock, kLevelOk}});
  s.run(8.0, {{kLock, kLevelWarn}});
  EXPECT_TRUE(s.health.acting());
  EXPECT_FALSE(s.cancels.empty());
}

TEST(LocalizationHealth, FirstCancelImmediateThenEveryRecancelPeriod)
{
  Sim s(params());
  s.run(2.0, {{kLock, kLevelOk}});
  s.run(16.0, {{kLock, kLevelError}});
  ASSERT_EQ(s.became_lost.size(), 1u);
  ASSERT_GE(s.cancels.size(), 4u);
  // On the same check that first declared LOST: the bug this guards made the
  // watchdog detect LOST and then never cancel at all.
  EXPECT_DOUBLE_EQ(s.cancels[0], s.became_lost[0]);
  for (std::size_t i = 1; i < s.cancels.size(); ++i) {
    EXPECT_NEAR(s.cancels[i] - s.cancels[i - 1], 3.0, 1e-6);
  }
}

TEST(LocalizationHealth, FirstSightingAlreadyErrorStartsTheClock)
{
  Sim s(params());
  s.run(20.0);  // a long silence before anything reports
  const double first = s.t;
  s.run(4.0, {{kLock, kLevelError}});
  EXPECT_TRUE(s.cancels.empty()) << "cancelled before lost_duration of observed ERROR";
  s.run(3.0, {{kLock, kLevelError}});
  ASSERT_FALSE(s.cancels.empty());
  EXPECT_GE(s.cancels[0], first + 5.0 - 1e-9);
}

TEST(LocalizationHealth, CancelGoalsFalseDetectsButNeverCancels)
{
  auto p = params();
  p.cancel_goals = false;
  Sim s(p);
  s.run(2.0, {{kLock, kLevelOk}});
  s.run(10.0, {{kLock, kLevelError}});
  EXPECT_TRUE(s.health.acting());
  EXPECT_TRUE(s.cancels.empty());
}

TEST(LocalizationHealth, RecoveryIsRequestedOncePerEpisode)
{
  auto p = params();
  p.call_recovery = true;
  LocalizationHealth h(p);
  int requests = 0;
  for (int i = 0; i <= 20; ++i) {  // ERROR from t = 0, checked every second
    h.report(kLock, kLevelError, i);
    requests += h.step(i).request_recovery;
  }
  EXPECT_EQ(requests, 1);
  h.report(kLock, kLevelOk, 21.0);
  EXPECT_TRUE(h.step(21.0).recovered);
  for (int i = 22; i <= 40; ++i) {  // a second episode
    h.report(kLock, kLevelError, i);
    requests += h.step(i).request_recovery;
  }
  EXPECT_EQ(requests, 2);
}

TEST(LocalizationHealth, UnwatchedStatusIsIgnored)
{
  LocalizationHealth h(params());
  h.report("some_other_node: status", kLevelError, 0.0);
  EXPECT_FALSE(h.seen("some_other_node: status"));
  EXPECT_FALSE(h.step(10.0).cancel_goals);
}
