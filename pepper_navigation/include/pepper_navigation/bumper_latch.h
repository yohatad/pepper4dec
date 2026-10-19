/* bumper_latch.h
 *
 * Latch and direction logic for bumper_guard, kept free of ROS types so it
 * can be tested directly (see test/test_bumper_latch.cpp).
 *
 * Why a latch and not a pass-through: /bumper is an EVENT topic -- naoqi
 * registers RightBumperPressed / LeftBumperPressed / BackBumperPressed and
 * publishes only on edges. Gating on the raw state chatters at contact: the
 * robot bumps, rebounds a millimetre, the switch releases, the gate opens,
 * MPPI drives forward again. So a press latches, and the latch outlives the
 * release by clear_delay.
 *
 * Why directional and not a stop: the controller runs vx_min 0.0, so MPPI can
 * never reverse. Zeroing cmd_vel on contact leaves Pepper leaning on the
 * obstacle with no way off it. Blocking only the direction that caused the
 * contact lets nav2_behaviors/BackUp -- the one thing in the stack that emits
 * negative vx -- drive out.
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

#ifndef PEPPER_NAVIGATION__BUMPER_LATCH_H_
#define PEPPER_NAVIGATION__BUMPER_LATCH_H_

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

/// Which translation directions the current latches forbid.
struct Clamp
{
  bool block_forward = false;
  bool block_reverse = false;
};

/// Sentinel for "no such event yet"; any real clock reading exceeds it.
constexpr double kNever = -1e30;

class BumperLatch
{
public:
  BumperLatch(double clear_delay, double stuck_timeout)
  : clear_delay_(clear_delay), stuck_timeout_(stuck_timeout) {}

  /// Apply one /bumper event. Out-of-range ids are ignored rather than
  /// trusted, since the id comes off the wire.
  void set(int id, bool pressed, double now_s)
  {
    if (id < 0 || id >= kBumperCount) {return;}
    State & s = st_[id];
    if (pressed) {
      if (!s.pressed) {s.pressed_since = now_s;}
      s.pressed = true;
      s.released_at = kNever;
      s.armed = true;
    } else if (s.pressed) {
      s.pressed = false;
      s.pressed_since = kNever;
      s.released_at = now_s;
    }
  }

  /// Operator override. Refuses while a switch is still physically pressed --
  /// clearing then would hand authority back while Pepper leans on the
  /// obstacle. Returns true when every latch was dropped.
  bool clear()
  {
    for (std::size_t i = 0; i < kBumperCount; ++i) {
      if (st_[i].pressed) {return false;}
    }
    for (std::size_t i = 0; i < kBumperCount; ++i) {
      st_[i].armed = false;
      st_[i].released_at = kNever;
    }
    return true;
  }

  /// Liveness gate. /bumper has no heartbeat, so silence is only safe to read
  /// as "not pressed" while the driver is known alive; blind, a latch holds
  /// instead of ageing out.
  void set_blind(bool blind) {blind_ = blind;}
  bool blind() const {return blind_;}

  bool latched(int id, double now_s) const
  {
    if (id < 0 || id >= kBumperCount) {return false;}
    const State & s = st_[id];
    if (s.pressed) {return true;}
    if (!s.armed) {return false;}
    if (blind_) {return true;}
    return (now_s - s.released_at) < clear_delay_;
  }

  Clamp clamp(double now_s) const
  {
    Clamp c;
    c.block_forward = latched(kRight, now_s) || latched(kLeft, now_s);
    c.block_reverse = latched(kBack, now_s);
    return c;
  }

  /// A switch held down far longer than a bump lasts -- debris, or a fault.
  /// Reported rather than silently immobilising the robot.
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

private:
  struct State
  {
    bool pressed = false;        // switch is down right now
    bool armed = false;          // pressed at some point since the last clear
    double pressed_since = kNever;
    double released_at = kNever;
  };

  State st_[kBumperCount];
  double clear_delay_;
  double stuck_timeout_;
  bool blind_ = false;
};

}  // namespace pepper_navigation

#endif  // PEPPER_NAVIGATION__BUMPER_LATCH_H_
