#!/usr/bin/env python
"""Put Pepper into a natural, still pose for navigation. Runs ON the robot.

Arms down at the Stand posture and HELD there at full stiffness; the three
things that move the arms during a walk switched off; the head homed once.

Why this shape, from measurements on 2026-10-21:

  * Stiffness 1.0 -- but asking for it is not enough. wakeUp() left HeadYaw at
    0.108 and HeadPitch at 0.329 APPLIED hardness. Two ALMotion features lower
    applied hardness below the stiffness you requested:
      - Smart Stiffness   reduces it on lightly loaded joints to save heat
      - Diagnosis Effect  reduces it on a joint reporting a fault, e.g. heat
    Neither is an Autonomous Ability, so Autonomous Life being "disabled" does
    NOT cover them -- they have to be switched off explicitly, before the
    setStiffnesses call, or the pose still is not held.

    MEASURED 2026-10-22, cold robot, all joints at 28-32 C, stiffness 1.0
    commanded on every one of them:
        HeadYaw        hardness 0.116      HipPitch   hardness 1.000
        HeadPitch               0.325      KneePitch           1.000
        shoulders         0.402 - 0.411
    Full hardness on exactly the two joints that carry the torso, a deep cut on
    everything unloaded: that is Smart Stiffness' signature, and at 28 C it
    cannot be Diagnosis Effect. So this script disables Smart Stiffness only.
    Diagnosis Effect stays ON: it is the backstop that cuts hardness on a joint
    in trouble, and RShoulderRoll reached 66 C on 2026-10-21. Removing a
    thermal protection to hold a pose is the wrong trade; --no-diagnosis-effect
    exists if a test needs it.

    Because Smart Stiffness is itself a heat-saving feature, the ORDER below
    matters: arms down first, then full stiffness. Holding a stretched arm at
    hardness 1.0 with Smart Stiffness off is how a shoulder cooks. A joint at partial
    hardness cannot hold its target, sags, and the controller then tracks the
    sag: that is the slow one-way creep. Commanded and sensed angles were equal
    to five decimals while HeadPitch walked -0.4755 -> -0.4326 -> -0.3789, so
    the joint was being actively re-targeted, not left loose.

  * NOT stiffness 0. Limp arms dangle and swing while the base moves, and
    /joint_angles gestures silently no-op, because the driver never calls
    setStiffnesses anywhere (checked all of src/subscribers/).

  * Arms DOWN, not StandZero. Arms-down measured 34-38 C through a run.
    ShoulderPitch 0 (StandZero) holds the arms horizontal, the longest moment
    arm there is; that is the pose that cooked RShoulderRoll to 66 C.

  * The three independent arm movers, each with its own switch:
      - gait swing          setMoveArmsEnabled    (the only limb switch in the
                            locomotion API; arm motion under move() is
                            documented and expected)
      - obstacle repositioning
                            setExternalCollisionProtectionEnabled("Arms")
                            <- this is the one that stretched the right arm
                            out during navigation. "Move" is a DIFFERENT
                            category and does not cover the arms.
      - idle breathing      setBreathEnabled("Arms"/"Head")

  * The movement Autonomous Abilities (Speaking/Listening/Background Movement,
    BasicAwareness) are gated by Autonomous Life, which is already "disabled",
    so they are inert whatever their flags say. They are set below only so this
    still behaves if Life is ever turned back on.

  * Volume and all of the above revert on a NAOqi restart, so this script is
    the thing to re-run, not a one-off set of qicli calls.

Usage:  python nav_ready.py [--volume 20] [--no-head-home]
"""
from __future__ import print_function

import argparse
import sys
import time

sys.path.insert(0, "/opt/aldebaran/lib/python2.7/site-packages")

import qi  # noqa: E402  (path has to be set first)


HELD = ["LArm", "RArm", "Head"]
REPORT = ["LShoulderPitch", "LShoulderRoll", "RShoulderPitch", "RShoulderRoll",
          "HeadYaw", "HeadPitch"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--volume", type=int, default=20,
                   help="ALAudioDevice master volume 0-100 (reverts on restart)")
    p.add_argument("--no-head-home", action="store_true",
                   help="leave the head wherever it is")
    p.add_argument("--stiffness", type=float, default=1.0,
                   help="arm/head stiffness; 1.0 holds the pose, 0 lets it dangle")
    p.add_argument("--no-diagnosis-effect", action="store_true",
                   help="also disable Diagnosis Effect (removes a thermal backstop)")
    args = p.parse_args()

    app = qi.Application(["nav_ready", "--qi-url=tcp://127.0.0.1:9559"])
    app.start()
    s = app.session
    m = s.service("ALMotion")
    posture = s.service("ALRobotPosture")
    mem = s.service("ALMemory")

    print("wakeUp")
    m.wakeUp()
    print("goToPosture Stand (arms down, near equilibrium)")
    posture.goToPosture("Stand", 0.5)
    time.sleep(1.0)

    # Must precede setStiffnesses: it caps the APPLIED hardness below the
    # stiffness requested, and it is not gated by Autonomous Life.
    methods = ["setSmartStiffnessEnabled"]
    if args.no_diagnosis_effect:
        methods.append("setDiagnosisEffectEnabled")
    for meth in methods:
        try:
            getattr(m, meth)(False)
            print("%s(False)" % meth)
        except Exception as exc:                                  # noqa: B902
            print("%s: %s" % (meth, exc))

    print("setStiffnesses %s = %.2f" % (HELD, args.stiffness))
    m.setStiffnesses(HELD, args.stiffness)

    print("breath off (Arms, Head)")
    m.setBreathEnabled("Arms", False)
    m.setBreathEnabled("Head", False)
    print("setMoveArmsEnabled(False, False)")
    m.setMoveArmsEnabled(False, False)
    print('setExternalCollisionProtectionEnabled("Arms", False)')
    m.setExternalCollisionProtectionEnabled("Arms", False)

    # Belt and braces: already inert while Autonomous Life is disabled.
    for name in ["ALSpeakingMovement", "ALListeningMovement",
                 "ALBackgroundMovement"]:
        try:
            s.service(name).setEnabled(False)
            print("%s disabled" % name)
        except Exception as exc:                                  # noqa: B902
            print("%s: %s" % (name, exc))

    if not args.no_head_home:
        print("head home (one shot)")
        m.setAngles(["HeadYaw", "HeadPitch"], [0.0, 0.0], 0.15)

    try:
        audio = s.service("ALAudioDevice")
        audio.setOutputVolume(args.volume)
        s.service("ALTextToSpeech").setVolume(1.0)
        print("volume %d%% (master), TTS multiplier 1.0" % args.volume)
    except Exception as exc:                                      # noqa: B902
        print("volume: %s" % exc)

    time.sleep(2.0)

    print("\n-- verification --")
    print("  moveArmsEnabled(Arms)   %s" % m.getMoveArmsEnabled("Arms"))
    print("  breathEnabled(Arms)     %s" % m.getBreathEnabled("Arms"))
    print("  breathEnabled(Head)     %s" % m.getBreathEnabled("Head"))
    print("  extCollisionProt(Arms)  %s"
          % m.getExternalCollisionProtectionEnabled("Arms"))
    for label, meth in [("smartStiffness", "getSmartStiffnessEnabled"),
                        ("diagnosisEffect", "getDiagnosisEffectEnabled")]:
        try:
            print("  %-22s  %s" % (label, getattr(m, meth)()))
        except Exception as exc:                                  # noqa: B902
            print("  %-22s  %s" % (label, exc))
    stiff = m.getStiffnesses(REPORT)
    ang = m.getAngles(REPORT, True)
    print("  %-18s %8s %8s %8s %7s" % (
        "joint", "stiff", "hardness", "angle", "temp"))
    bad = []
    for i, j in enumerate(REPORT):
        temp = mem.getData(
            "Device/SubDeviceList/%s/Temperature/Sensor/Value" % j)
        hard = mem.getData(
            "Device/SubDeviceList/%s/Hardness/Actuator/Value" % j)
        print("  %-18s %8.3f %8.3f %8.3f %6.1fC" % (j, stiff[i], hard, ang[i], temp))
        # Applied hardness is what decides whether the pose is held. A gap here
        # means something still overrides the request and the creep will remain.
        if args.stiffness - hard > 0.10:
            bad.append("%s %.3f" % (j, hard))
    if bad:
        print("\n  !! applied hardness still below the request: %s"
              % ", ".join(bad))
        print("  !! the pose will creep; see the 'Robot test: what moves the arms and head"
              " during move' item on github.com/users/yohatad/projects/2")
    print("\nRe-run this after every NAOqi restart.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
