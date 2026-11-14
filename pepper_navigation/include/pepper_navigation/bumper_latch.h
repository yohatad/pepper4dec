/* bumper_latch.h
 *
 * Latch logic for bumper_guard, kept free of ROS types so it can be tested
 * directly (see test/test_bumper_latch.cpp).
 *
 * A press means Pepper has already hit something, or an operator pressed a
 * bumper as an emergency stop. Either way the robot must not move again on its
 * own -- not forward, not back, not turning -- so any press latches a full stop
 * that only clear() lifts (bumper_guard's ~/clear service). clear() refuses
 * while a switch is still down, so a reset cannot hand authority back while
 * Pepper leans on the obstacle.
 *
 * Why a latch and not a pass-through: /bumper is an EVENT topic -- naoqi
 * registers RightBumperPressed / LeftBumperPressed / BackBumperPressed and
 * publishes only on edges, and the switch bounces (a second press ~50 ms after
 * the release, seen on the robot 2026-11-13).
 *
 * This replaced a directional latch (block only the side that was hit, release
 * 1 s after the switch) on 2026-11-13: with that, Nav2 could drive straight
 * back into the same obstacle, and a press could not serve as an emergency
 * stop. The cost is that a bump needs a person to reset it.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: October 19, 2026
 * Version: v2.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#pragma once

#include <cstddef>

namespace pepper_navigation
{

/// naoqi_bridge_msgs/Bumper::bumper values. BACK exists on Pepper only.
enum BumperId
{
  kRight = 0,
  kLeft = 1,
  kBack = 2,
  kBumperCount = 3
};

/// Sentinel for "no such event yet"; any real clock reading exceeds it.
constexpr double kNever = -1e30;

class BumperLatch
{
public:
  explicit BumperLatch(double stuck_timeout)
  : stuck_timeout_(stuck_timeout) {}

  /// Apply one /bumper event. Out-of-range ids are ignored rather than
  /// trusted, since the id comes off the wire.
  void set(int id, bool pressed, double now_s)
  {
    if (id < 0 || id >= kBumperCount) {return;}
    State & s = st_[id];
    if (pressed) {
      if (!s.pressed) {s.pressed_since = now_s;}
      s.pressed = true;
      s.armed = true;
    } else {
      s.pressed = false;
      s.pressed_since = kNever;
    }
  }

  /// Operator reset. Refuses while a switch is still physically pressed.
  /// Returns true when every latch was dropped.
  bool clear()
  {
    if (any_pressed()) {return false;}
    for (std::size_t i = 0; i < kBumperCount; ++i) {st_[i].armed = false;}
    return true;
  }

  /// joint_states liveness, reported in diagnostics: while it is absent,
  /// /bumper's silence proves nothing. It no longer changes the latch, which
  /// holds until a reset either way.
  void set_blind(bool blind) {blind_ = blind;}
  bool blind() const {return blind_;}

  /// This bumper was pressed since the last clear(), or is pressed now.
  bool latched(int id) const
  {
    if (id < 0 || id >= kBumperCount) {return false;}
    return st_[id].pressed || st_[id].armed;
  }

  /// Any latch set: hold every axis at zero.
  bool stop() const
  {
    for (int i = 0; i < kBumperCount; ++i) {
      if (latched(i)) {return true;}
    }
    return false;
  }

  /// A switch held down far longer than a bump lasts -- debris, or a fault.
  bool stuck(double now_s) const
  {
    for (std::size_t i = 0; i < kBumperCount; ++i) {
      if (st_[i].pressed && (now_s - st_[i].pressed_since) > stuck_timeout_) {return true;}
    }
    return false;
  }

  bool pressed(int id) const
  {
    if (id < 0 || id >= kBumperCount) {return false;}
    return st_[id].pressed;
  }

  bool any_pressed() const
  {
    for (std::size_t i = 0; i < kBumperCount; ++i) {
      if (st_[i].pressed) {return true;}
    }
    return false;
  }

private:
  struct State
  {
    bool pressed = false;        // switch is down right now
    bool armed = false;          // pressed at some point since the last clear
    double pressed_since = kNever;
  };

  State st_[kBumperCount];
  double stuck_timeout_;
  bool blind_ = false;
};

}  // namespace pepper_navigation
