#!/usr/bin/env python
"""Re-apply the settings Pepper forgets on every NAOqi start.

Loaded by NAOqi itself through /home/nao/naoqi/preferences/autoload.ini:

    [python]
    /home/nao/pepper_boot_defaults.py

Why this file exists: these values do not persist across a NAOqi restart or a
reboot, and re-applying them by hand after every boot was the actual daily
chore. Measured reverting on 2026-10-22 after a reboot: moveArmsEnabled back to
True/True, breathEnabled("Arms") back to True, the master volume back to 50,
and the web page pointer back to a wizard page.

WHAT IS DELIBERATELY *NOT* HERE
-------------------------------
Nothing in this file commands motion or removes a safeguard, because it runs
unattended at every boot:

  * no wakeUp, no goToPosture, no setStiffnesses -- the robot must not stand up
    or stiffen on its own when someone powers it on. The one motion-related
    call is ALAutonomousLife.setState("disabled"), only on a boot where the
    DisableLifeAndDialog preference had been lost: it puts the robot to rest.
  * no setExternalCollisionProtectionEnabled("Arms", False) -- that is a
    collision safeguard. It is off only for a navigation run, from nav_ready.py.
  * no setSmartStiffnessEnabled(False) -- Smart Stiffness is a heat-limiting
    feature. Disabling it permanently invites the 66 C RShoulderRoll episode of
    2026-10-21. nav_ready.py turns it off deliberately, with the arms already
    down, for the duration of a run.

Run `python nav_ready.py` before a navigation session for those.

The web page pointer is the one that is easy to lose: the robot's web app
rewrites CurrentPage to a wizard page on every "Next" click (settings.js:310),
and the wizard cannot be completed any more -- its step 4 contacts the
Aldebaran cloud, which shut down 2026-10-01. Setting it to a menu.* page makes
verifyTransition's `cloudState == "menu"` branch whitelist every state, which is
the only way into the normal menu.
"""
from __future__ import print_function

import sys
import threading
import time

sys.path.insert(0, "/opt/aldebaran/lib/python2.7/site-packages")

LOG_PATH = "/home/nao/pepper_boot_defaults.log"

# (domain, key, value) -- strings only; ALPreferenceManager keeps the type.
PREFS = [
    ("com.aldebaran.robotwebpage", "CurrentPage", "menu.root.myrobot"),
    ("com.aldebaran.settings", "InteractiveConfigModality", "NoInteractive"),
    # Keeps Autonomous Life off at boot, so Pepper stays at rest. NAOqi
    # sometimes judges prefs.db corrupted at boot (Corrupted-*.db, the files
    # pass sqlite's integrity_check) and starts an empty one; then Life comes
    # up "solitary" and wakes the robot. Seen 2026-11-27.
    ("com.aldebaran.debug", "DisableLifeAndDialog", "1"),
    ("com.aldebaran.robotwebpage", "EulaVersion", "2.9"),
]

LIFE_PREF = ("com.aldebaran.debug", "DisableLifeAndDialog")
LIFE_SETTLE_TIMEOUT = 120.0   # how long to wait for Life to start, if it does

VOLUME = 20                 # ALAudioDevice master, 0-100
CONNECT_TIMEOUT = 240.0     # NAOqi is still starting when this module loads
RETRY_EVERY = 3.0


def log(msg):
    line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    try:
        fh = open(LOG_PATH, "a")
        try:
            fh.write(line + "\n")
        finally:
            fh.close()
    except Exception:                                             # noqa: B902
        pass
    print("[pepper_boot_defaults] %s" % msg)


def connect():
    """Wait for the service directory. This module loads before it is up."""
    import qi
    deadline = time.time() + CONNECT_TIMEOUT
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            session = qi.Session()
            session.connect("tcp://127.0.0.1:9559")
            log("connected on attempt %d" % attempt)
            return session
        except Exception as exc:                                  # noqa: B902
            if attempt == 1 or attempt % 10 == 0:
                log("waiting for NAOqi (attempt %d): %s" % (attempt, exc))
            time.sleep(RETRY_EVERY)
    log("GAVE UP after %.0f s" % CONNECT_TIMEOUT)
    return None


def disable_life_when_it_starts(session):
    """Turn Autonomous Life off if it starts on this boot.

    Only needed when DisableLifeAndDialog had been lost: Life read the empty
    store before this module restored it. Setting Life to "disabled" puts the
    robot to rest -- the direction this file allows; it never wakes it.
    """
    try:
        life = session.service("ALAutonomousLife")
    except Exception as exc:                                      # noqa: B902
        log("ALAutonomousLife unavailable: %s" % exc)
        return
    deadline = time.time() + LIFE_SETTLE_TIMEOUT
    while time.time() < deadline:
        try:
            state = life.getState()
            if state != "disabled":
                life.setState("disabled")
                log("Autonomous Life was %r on boot -> %r"
                    % (state, life.getState()))
                return
        except Exception as exc:                                  # noqa: B902
            log("Autonomous Life check FAILED: %s" % exc)
            return
        time.sleep(RETRY_EVERY)
    log("Autonomous Life stayed disabled")


def apply_all(session):
    life_pref_restored = False
    try:
        pref = session.service("ALPreferenceManager")
        for domain, key, value in PREFS:
            try:
                before = pref.getValue(domain, key)
                if before == value:
                    log("%s/%s already %r" % (domain, key, value))
                    continue
                if (domain, key) == LIFE_PREF:
                    life_pref_restored = True
                # setValue keeps a stored value's type; a value stored
                # wrongly (for example with extra quotes) is replaced.
                if before is not None:
                    pref.removeValue(domain, key)
                pref.setValue(domain, key, value)
                log("%s/%s  %r -> %r" % (domain, key, before,
                                         pref.getValue(domain, key)))
            except Exception as exc:                              # noqa: B902
                log("%s/%s FAILED: %s" % (domain, key, exc))
    except Exception as exc:                                      # noqa: B902
        log("ALPreferenceManager unavailable: %s" % exc)

    try:
        session.service("ALAudioDevice").setOutputVolume(VOLUME)
        session.service("ALTextToSpeech").setVolume(1.0)
        log("volume %d%% master, TTS multiplier 1.0" % VOLUME)
    except Exception as exc:                                      # noqa: B902
        log("volume FAILED: %s" % exc)

    # Stops the arms and hands fidgeting. Commands no motion and removes no
    # safeguard: breath is an animation, and MoveArms only governs gait swing.
    try:
        motion = session.service("ALMotion")
        for chain in ["Arms", "Head"]:
            motion.setBreathEnabled(chain, False)
        motion.setMoveArmsEnabled(False, False)
        log("breath Arms/Head off, moveArmsEnabled(Arms)=%s"
            % motion.getMoveArmsEnabled("Arms"))
    except Exception as exc:                                      # noqa: B902
        log("ALMotion settings FAILED: %s" % exc)

    if life_pref_restored:
        disable_life_when_it_starts(session)


def worker():
    log("--- start (pid %d) ---" % _getpid())
    session = connect()
    if session is not None:
        apply_all(session)
    log("--- done ---")


def _getpid():
    try:
        import os
        return os.getpid()
    except Exception:                                             # noqa: B902
        return -1


# Must never raise at import: NAOqi loads this during its own startup, and a
# traceback here would surface as a NAOqi module load failure. A daemon thread
# also keeps the retry loop from blocking that startup.
try:
    _t = threading.Thread(target=worker)
    _t.daemon = True
    _t.start()
except Exception:                                                 # noqa: B902
    log("could not start thread")

if __name__ == "__main__":
    # Also runnable by hand, for testing: python pepper_boot_defaults.py
    _t.join(CONNECT_TIMEOUT + 30.0)
