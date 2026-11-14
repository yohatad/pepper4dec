#!/usr/bin/env python
"""Find out what moves Pepper's arms and head during ALMotion.move, without ROS.

Runs ON the robot (python 2.7, NAOqi 2.5). Nothing here touches ROS: the point
is to drive ALMotion directly so that nav2, the driver and the collision
monitor are all out of the picture, and any arm or head motion that remains is
NAOqi's own.

Phases, each answering one question (see the 'Robot test: what moves the arms
and head during move' item on github.com/users/yohatad/projects/2):

  dump      what is enabled right now? read-only, no motion, no preparation.
  idle      do the arms/head drift with NO move call at all?
  moveinit  does moveInit() alone re-pose the arms/head?
  single    what does ONE latched move() call do?
  stream    does calling move() at 20 Hz (nav2's rate) differ from once?
  chatter   does repeated zero -> non-zero (nav2 + collision monitor) make the
            excursion grow, i.e. does each restart re-pose the arms?
  stopcmp   how far does the robot roll on after move(0,0,0) -- what the
            driver sends -- versus stopMove()? The 2.4 docs say stopMove ends
            the task "more quickly than move(0.0, 0.0, 0.0)", and the driver
            uses the slow one in both its watchdog and its proximity stop.

What the 2.4 locomotion API does and does not say (doc.aldebaran.com is dead;
copy of the page read 2026-10-21):
  - move() is explicitly non-blocking, and the last velocity command wins. The
    page does NOT say move() calls moveInit itself; that is this test's
    hypothesis, not documented behaviour.
  - moveInit() "checks the robot pose and takes a right posture", blocking.
  - setMoveArmsEnabled is the ONLY limb switch in the whole locomotion API, and
    nothing in it touches the head. So arm motion under move() is expected and
    documented; head motion under move() is not, and must come from elsewhere
    (breath/background/listening movement) -- which is what `dump` checks.

Usage on the robot:

    python move_probe.py dump
    python move_probe.py idle --duration 60
    python move_probe.py single --duration 10 --vx 0.08
    python move_probe.py stream --duration 10 --vx 0.08
    python move_probe.py chatter --cycles 5
    python move_probe.py all            # dump, idle, moveinit, single, stream, chatter

Every phase writes a CSV and prints a per-joint excursion/drift/temperature
summary. Ctrl-C stops the base and still writes the summary.
"""
from __future__ import print_function

import argparse
import csv
import datetime
import os
import sys
import time

sys.path.insert(0, "/opt/aldebaran/lib/python2.7/site-packages")

import qi  # noqa: E402  (path has to be set first)


ARM_JOINTS = [
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll",
]
HEAD_JOINTS = ["HeadYaw", "HeadPitch"]
# Context only: these two hold the torso up and were the other warm joints.
BODY_JOINTS = ["HipRoll", "HipPitch", "KneePitch"]

WATCHED = ARM_JOINTS + HEAD_JOINTS + BODY_JOINTS

# Joints whose temperature aborts the test. The 2026-10-21 episode cooked
# RShoulderRoll to 66 C, so the shoulders are the ones to watch.
ABORT_JOINTS = [
    "LShoulderPitch", "LShoulderRoll", "RShoulderPitch", "RShoulderRoll",
    "HeadYaw", "HeadPitch",
]

MEM_TEMP = "Device/SubDeviceList/%s/Temperature/Sensor/Value"
MEM_TSTATUS = "Device/SubDeviceList/%s/Temperature/Status/Value"
MEM_CURRENT = "Device/SubDeviceList/%s/ElectricCurrent/Sensor/Value"
MEM_HARDNESS = "Device/SubDeviceList/%s/Hardness/Actuator/Value"


class Abort(Exception):
    """Raised when a joint gets too hot to keep going."""


class Probe(object):

    def __init__(self, args):
        self.args = args
        self.app = qi.Application(["move_probe", "--qi-url=tcp://127.0.0.1:9559"])
        self.app.start()
        self.session = self.app.session
        self.motion = self.session.service("ALMotion")
        self.posture = self.session.service("ALRobotPosture")
        self.memory = self.session.service("ALMemory")
        self.rows = []
        self.keys_temp = [MEM_TEMP % j for j in WATCHED]
        self.keys_tstat = [MEM_TSTATUS % j for j in ABORT_JOINTS]
        self.keys_curr = [MEM_CURRENT % j for j in WATCHED]
        self.keys_hard = [MEM_HARDNESS % j for j in WATCHED]
        self.t0 = time.time()

    # -- helpers ----------------------------------------------------------

    def svc(self, name):
        """Return a service or None; several of these are absent on this head."""
        try:
            return self.session.service(name)
        except Exception as exc:                                  # noqa: B902
            print("  %-24s UNAVAILABLE (%s)" % (name, type(exc).__name__))
            return None

    @staticmethod
    def try_call(obj, names, *a):
        """Call the first method in `names` that exists and accepts `a`.

        NAOqi's getter names are inconsistent (isEnabled vs getEnabled) and
        qicli only reveals the real signature on an arity mismatch, so probe
        rather than guess. Returns (value, method_name) or (error_str, None).
        """
        if obj is None:
            return ("no service", None)
        last = "no such method"
        for n in names:
            try:
                return (getattr(obj, n)(*a), n)
            except Exception as exc:                              # noqa: B902
                last = "%s: %s" % (n, exc)
        return (last, None)

    def sample(self, phase, note=""):
        """One row per joint: commanded vs sensed angle, stiffness, temp, current."""
        t = time.time() - self.t0
        cmd = self.motion.getAngles(WATCHED, False)
        sen = self.motion.getAngles(WATCHED, True)
        stiff = self.motion.getStiffnesses(WATCHED)
        temp = self.memory.getListData(self.keys_temp)
        curr = self.memory.getListData(self.keys_curr)
        hard = self.memory.getListData(self.keys_hard)
        try:
            active = 1 if self.motion.moveIsActive() else 0
        except Exception:                                         # noqa: B902
            active = -1
        for i, j in enumerate(WATCHED):
            self.rows.append({
                "phase": phase, "t": round(t, 3), "joint": j,
                "cmd": cmd[i], "sensor": sen[i],
                "err": cmd[i] - sen[i],
                "stiffness": stiff[i], "hardness": hard[i],
                "temp": temp[i], "current": curr[i],
                "move_active": active, "note": note,
            })
        # Thermal guard. Status >= 1 is NAOqi's own "hot" flag.
        hot = []
        tstat = self.memory.getListData(self.keys_tstat)
        for i, j in enumerate(ABORT_JOINTS):
            k = WATCHED.index(j)
            if temp[k] >= self.args.temp_abort or (tstat[i] or 0) >= 1:
                hot.append("%s %.1fC status=%s" % (j, temp[k], tstat[i]))
        if hot:
            raise Abort("; ".join(hot))

    def watch(self, phase, seconds, tick=None, note=""):
        """Sample at --rate for `seconds`, calling tick(elapsed) each loop."""
        end = time.time() + seconds
        nxt = time.time()
        period = 1.0 / self.args.rate
        while time.time() < end:
            now = time.time()
            if now >= nxt:
                self.sample(phase, note)
                nxt = now + period
            if tick is not None:
                tick(now - (end - seconds))
            else:
                time.sleep(0.02)

    def countdown(self, what):
        if self.args.yes:
            return
        print("\n  !! %s" % what)
        print("  !! Needs ~2 m of clear floor ahead and the driver/nav2 STOPPED.")
        for i in range(5, 0, -1):
            sys.stdout.write("\r  starting in %d (Ctrl-C to abort) " % i)
            sys.stdout.flush()
            time.sleep(1)
        print("")

    def stop_base(self):
        """Safety stop: stopMove() first because it is the quicker of the two.

        move(0,0,0) still follows, to clear the latched velocity so a later
        call cannot resume on it.
        """
        try:
            self.motion.stopMove()
            self.motion.move(0.0, 0.0, 0.0)
        except Exception as exc:                                  # noqa: B902
            print("  stop_base failed: %s" % exc)

    # -- preparation ------------------------------------------------------

    def prepare(self):
        """The nav_ready state: arms down, held at full stiffness, movers off.

        Deliberately explicit about stiffness: wakeUp() alone left HeadYaw at
        0.108 and HeadPitch at 0.329 on 2026-10-21, which is the suspected
        cause of the slow one-way creep.
        """
        print("  prepare: wakeUp + Stand + stiffness 1.0 + movers off")
        self.motion.wakeUp()
        self.posture.goToPosture("Stand", 0.5)
        time.sleep(1.0)
        # Smart Stiffness reduces applied hardness below the requested
        # stiffness, so it has to go off FIRST or setStiffnesses below does not
        # actually hold the pose. It is not gated by Autonomous Life.
        # Diagnosis Effect is left ON unless --no-diagnosis-effect: it is the
        # thermal backstop, and the 2026-10-22 dump showed it is not what caps
        # hardness on a cold robot.
        methods = ["setSmartStiffnessEnabled"]
        if self.args.no_diagnosis_effect:
            methods.append("setDiagnosisEffectEnabled")
        for meth in methods:
            val, used = self.try_call(self.motion, [meth], False)
            print("  prepare: %s(False) -> %s" % (meth, "ok" if used else val))
        self.motion.setStiffnesses(["LArm", "RArm", "Head"], 1.0)
        self.motion.setBreathEnabled("Arms", False)
        self.motion.setBreathEnabled("Head", False)
        self.motion.setMoveArmsEnabled(False, False)
        # "Arms" is the category that stretched the right arm out during
        # navigation; "Move" is a different one and does not cover it.
        self.motion.setExternalCollisionProtectionEnabled("Arms", False)
        sm = self.svc("ALSpeakingMovement")
        if sm is not None:
            self.try_call(sm, ["setEnabled"], False)
        self.motion.setAngles(HEAD_JOINTS, [0.0, 0.0], 0.15)
        time.sleep(2.0)
        print("  prepare: done, settling 3 s")
        time.sleep(3.0)

    # -- phases -----------------------------------------------------------

    def phase_dump(self):
        """Read-only audit. No motion, no preparation: report the live state."""
        print("\n=== dump: what is enabled right now ===")
        m = self.motion
        print("  robotIsWakeUp        %s" % m.robotIsWakeUp())
        print("  moveIsActive         %s" % m.moveIsActive())
        # "Arms" is also valid and returns true only if both are enabled.
        for chain in ["LArm", "RArm", "Arms"]:
            print("  moveArmsEnabled(%-4s) %s" % (chain, self.try_call(
                m, ["getMoveArmsEnabled"], chain)[0]))
        for chain in ["Body", "Legs", "Arms", "LArm", "RArm", "Head"]:
            print("  breathEnabled(%-4s)   %s   idlePosture %s" % (
                chain,
                self.try_call(m, ["getBreathEnabled"], chain)[0],
                self.try_call(m, ["getIdlePostureEnabled"], chain)[0]))
        for cat in ["All", "Move", "Arms", "LArm", "RArm"]:
            print("  extCollisionProt(%-4s) %s" % (cat, self.try_call(
                m, ["getExternalCollisionProtectionEnabled"], cat)[0]))
        print("  selfCollisionProt(Arms) %s" % self.try_call(
            m, ["getCollisionProtectionEnabled"], "Arms")[0])
        for which in ["Default", "Max"]:
            print("  moveConfig(%s) %s" % (which, self.try_call(
                m, ["getMoveConfig"], which)[0]))

        # ALMotion features that are NOT gated by Autonomous Life. These are
        # the live candidates: Smart Stiffness and Diagnosis Effect both LOWER
        # a joint's applied hardness below the stiffness you asked for, which
        # is exactly the 2026-10-21 measurement (stiffness 1.0 commanded,
        # HeadYaw hardness 0.108, HeadPitch 0.329) and the suspected cause of
        # the slow one-way creep.
        print("\n  -- ALMotion features, independent of Autonomous Life --")
        for label, names in [
                ("smartStiffness", ["getSmartStiffnessEnabled"]),
                ("diagnosisEffect", ["getDiagnosisEffectEnabled"]),
                ("fallManager", ["getFallManagerEnabled"]),
                ("pushRecovery", ["getPushRecoveryEnabled"]),
                ("smartStiffnessCfg", ["getMotionConfig"])]:
            val, meth = self.try_call(m, names)
            print("  %-22s %s%s" % (label, val, "" if meth else "  (no getter)"))

        # Gated by Autonomous Life: when getState is "disabled" these are all
        # already inert whatever their enabled flag says, so this block is
        # confirmation, not a list of suspects.
        print("\n  -- Autonomous Abilities (gated by ALAutonomousLife) --")
        life = self.svc("ALAutonomousLife")
        if life is not None:
            print("  %-22s state=%s" % (
                "ALAutonomousLife", self.try_call(life, ["getState"])[0]))
        for name in ["ALSpeakingMovement", "ALListeningMovement",
                     "ALBackgroundMovement", "ALAutonomousMoves"]:
            svc = self.svc(name)
            if svc is None:
                continue
            val, meth = self.try_call(svc, ["isEnabled", "getEnabled"])
            print("  %-22s flag=%s (via %s) -- inert while Life is disabled"
                  % (name, val, meth))
        for name, meths in [("ALTracker", ["isActive"]),
                            ("ALBasicAwareness", ["isEnabled", "isRunning"])]:
            svc = self.svc(name)
            if svc is not None:
                print("  %-22s %s" % (name, self.try_call(svc, meths)[0]))

        print("\n  -- stiffness / hardness / temperature now --")
        stiff = self.motion.getStiffnesses(WATCHED)
        hard = self.memory.getListData(self.keys_hard)
        temp = self.memory.getListData(self.keys_temp)
        print("  %-18s %8s %8s %7s  %s" % (
            "joint", "stiff", "hardness", "temp", "gap"))
        for i, j in enumerate(WATCHED):
            gap = stiff[i] - hard[i]
            # A gap means something is overriding the request: Smart Stiffness
            # or Diagnosis Effect. Stiffness 1.0 will not hold the pose.
            print("  %-18s %8.3f %8.3f %6.1fC  %s" % (
                j, stiff[i], hard[i], temp[i],
                "%+.3f  <-- OVERRIDDEN" % gap if abs(gap) > 0.10 else ""))
        # One sample so the CSV is never empty for this phase.
        self.sample("dump")

    def phase_idle(self):
        """Control: hold still, send nothing. Any drift here is not from move()."""
        print("\n=== idle: %.0f s, no move() at all ===" % self.args.duration)
        self.prepare()
        self.watch("idle", self.args.duration)

    def phase_moveinit(self):
        """Does the posture step alone move the arms/head?"""
        print("\n=== moveinit: moveInit() alone, no velocity ===")
        self.prepare()
        self.watch("moveinit_pre", 5.0)
        self.countdown("moveInit() may bend the knees and shift the stance.")
        # Documented as blocking: it returns once the posture is taken.
        self.motion.moveInit()
        self.watch("moveinit_post", self.args.duration, note="after moveInit")

    def phase_single(self):
        """ONE latched move() call, the way the API is meant to be used."""
        print("\n=== single: one move(%.3f, 0, %.3f), latched %.0f s ===" % (
            self.args.vx, self.args.vth, self.args.duration))
        self.prepare()
        self.watch("single_pre", 5.0)
        self.countdown("The robot will WALK FORWARD.")
        self.motion.move(self.args.vx, 0.0, self.args.vth)
        self.watch("single_move", self.args.duration, note="latched")
        # The driver's own stop, deliberately: see stopcmp.
        self.motion.move(0.0, 0.0, 0.0)
        self.watch("single_post", 8.0, note="stopped")

    def phase_stream(self):
        """nav2's steady state: move() re-sent every controller tick."""
        hz = self.args.cmd_rate
        print("\n=== stream: move() at %.0f Hz for %.0f s ===" % (hz, self.args.duration))
        self.prepare()
        self.watch("stream_pre", 5.0)
        self.countdown("The robot will WALK FORWARD (commands at %.0f Hz)." % hz)
        state = {"nxt": time.time(), "n": 0}

        def tick(elapsed):
            now = time.time()
            if now >= state["nxt"]:
                # A gentle ramp, like MPPI's output rather than a constant.
                frac = 0.6 + 0.4 * (1.0 + (elapsed % 4.0 - 2.0) / 2.0) / 2.0
                self.motion.move(self.args.vx * frac, 0.0, self.args.vth)
                state["nxt"] = now + 1.0 / hz
                state["n"] += 1
            time.sleep(0.005)

        self.watch("stream_move", self.args.duration, tick=tick)
        self.stop_base()
        print("  sent %d move() calls" % state["n"])
        self.watch("stream_post", 8.0, note="stopped")

    def phase_chatter(self):
        """The hypothesis: every zero -> non-zero restart re-poses the arms.

        nav2 plus the collision monitor produce exactly this. If the arm/head
        excursion grows cycle over cycle, the restart is the mechanism; if each
        cycle looks like the first, it is not.
        """
        hz = self.args.cmd_rate
        n = self.args.cycles
        print("\n=== chatter: %d x (%.0fs move @ %.0f Hz + %.0fs zeros) ===" % (
            n, self.args.on, hz, self.args.off))
        self.prepare()
        self.watch("chatter_pre", 5.0)
        self.countdown("The robot will WALK in %d short bursts." % n)
        for c in range(n):
            tag = "cycle%d" % (c + 1)
            state = {"nxt": time.time()}

            def tick(_elapsed, st=state):
                now = time.time()
                if now >= st["nxt"]:
                    self.motion.move(self.args.vx, 0.0, self.args.vth)
                    st["nxt"] = now + 1.0 / hz
                time.sleep(0.005)

            self.watch("chatter_on_" + tag, self.args.on, tick=tick)
            state2 = {"nxt": time.time()}

            def tick0(_elapsed, st=state2):
                now = time.time()
                if now >= st["nxt"]:
                    self.motion.move(0.0, 0.0, 0.0)
                    st["nxt"] = now + 1.0 / hz
                time.sleep(0.005)

            self.watch("chatter_off_" + tag, self.args.off, tick=tick0)
        self.stop_base()
        self.watch("chatter_post", 8.0, note="stopped")

    def phase_stopcmp(self):
        """Braking distance of move(0,0,0) vs stopMove(), measured.

        PolygonStop is a 0.40 m wedge and the driver stops with move(0,0,0).
        If that rolls further than stopMove(), the wedge is sized against the
        wrong number. Distance comes from getRobotPosition(True), i.e. odometry.
        """
        print("\n=== stopcmp: roll-on after each stop method ===")
        self.prepare()
        results = []
        for method in ["move_zero", "stopMove"]:
            self.countdown("The robot will WALK then stop (%s)." % method)
            self.motion.move(self.args.vx, 0.0, 0.0)
            # Let the gait reach steady state before braking.
            self.watch("stopcmp_%s_run" % method, 6.0, note=method)
            p0 = self.motion.getRobotPosition(True)
            t0 = time.time()
            if method == "move_zero":
                self.motion.move(0.0, 0.0, 0.0)
            else:
                self.motion.stopMove()
            while self.motion.moveIsActive() and time.time() - t0 < 5.0:
                time.sleep(0.02)
            settle = time.time() - t0
            time.sleep(1.5)
            p1 = self.motion.getRobotPosition(True)
            dist = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
            results.append((method, dist, settle))
            print("  %-10s rolled %.3f m, moveIsActive cleared after %.2f s"
                  % (method, dist, settle))
            self.watch("stopcmp_%s_post" % method, 5.0, note=method)
            self.stop_base()
            time.sleep(2.0)
        print("\n  -- braking distance at vx=%.2f m/s --" % self.args.vx)
        for method, dist, settle in results:
            print("  %-10s %.3f m  (%.2f s)" % (method, dist, settle))
        print("  PolygonStop reaches 0.40 m ahead of base_footprint.")

    # -- output -----------------------------------------------------------

    def write_csv(self, path):
        cols = ["phase", "t", "joint", "cmd", "sensor", "err", "stiffness",
                "hardness", "temp", "current", "move_active", "note"]
        if sys.version_info[0] >= 3:
            fh = open(path, "w")
        else:
            fh = open(path, "wb")
        try:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
        finally:
            fh.close()
        print("\n  wrote %s (%d rows)" % (path, len(self.rows)))

    def summary(self):
        """Per phase and joint: excursion, net drift, stiffness sag, temp rise."""
        print("\n=== summary (rad, degrees in brackets) ===")
        phases = []
        for r in self.rows:
            if r["phase"] not in phases:
                phases.append(r["phase"])
        for ph in phases:
            sub = [r for r in self.rows if r["phase"] == ph]
            if len(sub) < 2 * len(WATCHED):
                continue
            print("\n  -- %s (%.0f s) --" % (ph, sub[-1]["t"] - sub[0]["t"]))
            print("  %-16s %9s %9s %9s %9s %9s %8s" % (
                "joint", "excursion", "net", "net(deg)", "stiff", "hardness",
                "dTemp"))
            for j in WATCHED:
                js = [r for r in sub if r["joint"] == j]
                if not js:
                    continue
                first = js[0]["sensor"]
                vals = [r["sensor"] for r in js]
                exc = max(vals) - min(vals)
                net = js[-1]["sensor"] - first
                st = [r["stiffness"] for r in js]
                hd = [r["hardness"] for r in js]
                dt = js[-1]["temp"] - js[0]["temp"]
                flag = ""
                if abs(net) > 0.05:
                    flag = "  <-- DRIFT"
                elif exc > 0.10:
                    flag = "  <-- swing"
                if max(st) - min(hd) > 0.10:
                    flag += " [hardness < stiffness]"
                print("  %-16s %9.4f %9.4f %9.1f %4.2f-%4.2f %4.2f-%4.2f %+7.1f%s"
                      % (j, exc, net, net * 57.2958, min(st), max(st),
                         min(hd), max(hd), dt, flag))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("phase", nargs="+", choices=[
        "dump", "idle", "moveinit", "single", "stream", "chatter", "stopcmp",
        "all"])
    p.add_argument("--duration", type=float, default=15.0,
                   help="seconds of the moving/holding part of a phase")
    p.add_argument("--vx", type=float, default=0.08, help="m/s forward (capped 0.15)")
    p.add_argument("--vth", type=float, default=0.0, help="rad/s yaw")
    p.add_argument("--cmd-rate", type=float, default=20.0,
                   help="move() calls per second; 20 matches controller_frequency")
    p.add_argument("--rate", type=float, default=2.0, help="samples per second")
    p.add_argument("--cycles", type=int, default=5, help="chatter: burst count")
    p.add_argument("--on", type=float, default=2.0, help="chatter: seconds moving")
    p.add_argument("--off", type=float, default=1.0, help="chatter: seconds of zeros")
    p.add_argument("--temp-abort", type=float, default=55.0,
                   help="stop if a watched joint reaches this (C)")
    p.add_argument("--out", default=None, help="CSV path")
    p.add_argument("--no-diagnosis-effect", action="store_true",
                   help="also disable Diagnosis Effect (removes a thermal backstop)")
    p.add_argument("--yes", action="store_true", help="skip the walk countdowns")
    args = p.parse_args()

    args.vx = max(-0.15, min(0.15, args.vx))
    phases = ["dump", "idle", "moveinit", "single", "stream", "chatter"] \
        if "all" in args.phase else args.phase

    out = args.out or os.path.join(
        "/home/nao",
        "move_probe_%s.csv" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    probe = Probe(args)
    try:
        for ph in phases:
            getattr(probe, "phase_" + ph)()
    except Abort as exc:
        print("\n  !! THERMAL ABORT: %s" % exc)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        probe.stop_base()
        probe.write_csv(out)
        probe.summary()
        print("\n  base stopped. Stiffness left as-is; run nav_ready.py or rest().")
    return 0


if __name__ == "__main__":
    sys.exit(main())
