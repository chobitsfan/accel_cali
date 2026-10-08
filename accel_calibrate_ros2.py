#!/usr/bin/env python3
"""
Gauss-Newton accelerometer calibration driven by ROS 2 IMU data, with a
text UI.

This is a faithful Python port of ArduPilot's
libraries/AP_AccelCal/AccelCalibrator.cpp (Gauss-Newton least-squares fit of
accelerometer offset / scale / cross-axis parameters) plus the 6-position
orientation sequence used by libraries/AP_AccelCal/AP_AccelCal.cpp.

It subscribes to a sensor_msgs/msg/Imu topic, walks the operator through the
standard six (or more) vehicle orientations, averages the static samples over a
configurable window, then solves for the calibration parameters exactly the way
the firmware does.

Usage (ROS 2):
    python3 accel_calibrate_ros2.py --topic /imu --fit ellipsoid --samples 8

Usage (offline self-test, no ROS 2 needed):
    python3 accel_calibrate_ros2.py --simulate

Authored after Jonathan Challinger's original C++ AccelCalibrator.
"""

import argparse
import sys
import threading
import time

import numpy as np

GRAVITY_MSS = 9.80665

# fit types
FIT_AXIS_ALIGNED = 0      # 6 params: 3 offset + 3 diagonal (scale)
FIT_ELLIPSOID = 1         # 9 params: + 3 off-diagonal (cross-axis)

# calibrator status
CAL_NOT_STARTED = 0
CAL_WAITING_FOR_ORIENTATION = 1
CAL_COLLECTING_SAMPLE = 2
CAL_SUCCESS = 3
CAL_FAILED = 4

ACCEL_CAL_TOLERANCE = 0.1
MAX_ITERATIONS = 50

# Human-readable orientation prompts, matching AP_AccelCal.cpp ordering.
ORIENTATION_PROMPTS = [
    "Z down",
    "Y up",
    "Y down",
    "X down",
    "X up",
    "Z up",
    # extra positions for ellipsoid fits (>6 samples); operator picks any
    # distinct attitudes (e.g. tilted corners).
    "in a distinct TILTED attitude",
    "in another distinct TILTED attitude",
    "in yet another distinct TILTED attitude",
    "in a further distinct TILTED attitude",
]


def _build_M(p):
    """Build the 3x3 correction matrix from a 9-element parameter vector.

    Parameter layout (matches the C++ union): [ox,oy,oz, dx,dy,dz, odx,ody,odz]
    """
    dx, dy, dz = p[3], p[4], p[5]
    odx, ody, odz = p[6], p[7], p[8]
    return np.array([
        [dx,  odx, ody],
        [odx, dy,  odz],
        [ody, odz, dz],
    ])


class AccelCalibrator:
    """Port of AP_AccelCal's AccelCalibrator (Gauss-Newton ellipsoid fit)."""

    def __init__(self):
        self._tolerance = ACCEL_CAL_TOLERANCE
        self._status = CAL_NOT_STARTED
        self.clear()

    # ----- lifecycle ---------------------------------------------------
    def clear(self):
        self._status = CAL_NOT_STARTED
        self._samples_collected = 0
        # accumulators per sample slot: integrated delta-velocity and delta-time
        self._dvel = None
        self._dt = None
        self._param = np.zeros(9)
        self._fitness = 1.0e30
        self._min_sample_dist = 0.0

    def start(self, fit_type, num_samples, sample_time,
              offset=(0, 0, 0), diag=(1, 1, 1), offdiag=(0, 0, 0)):
        if self._status in (CAL_FAILED, CAL_SUCCESS):
            self.clear()
        if self._status != CAL_NOT_STARTED:
            return

        self._fit_type = fit_type
        self._num_samples = num_samples
        self._sample_time = sample_time

        # minimum angular separation between accepted samples, derived from a
        # regular polyhedron with `2*num_samples-4` triangular faces.
        faces = 2 * num_samples - 4
        a = (4.0 * np.pi / (3.0 * faces)) + np.pi / 3.0
        theta = 0.5 * np.arccos(np.cos(a) / (1.0 - np.cos(a)))
        self._min_sample_dist = GRAVITY_MSS * 2 * np.sin(theta / 2)

        self._param[0:3] = offset
        self._param[3:6] = diag
        self._param[6:9] = offdiag

        min_needed = 6 if fit_type == FIT_AXIS_ALIGNED else 8
        if num_samples < min_needed:
            self._status = CAL_FAILED
            return

        self._dvel = np.zeros((num_samples, 3))
        self._dt = np.zeros(num_samples)
        self._samples_collected = 0
        self._status = CAL_WAITING_FOR_ORIENTATION

    def collect_sample(self):
        if self._status == CAL_WAITING_FOR_ORIENTATION:
            self._status = CAL_COLLECTING_SAMPLE

    def running(self):
        return self._status in (CAL_WAITING_FOR_ORIENTATION, CAL_COLLECTING_SAMPLE)

    @property
    def status(self):
        return self._status

    @property
    def fitness(self):
        return self._fitness

    @property
    def samples_collected(self):
        return self._samples_collected

    # ----- sample ingestion -------------------------------------------
    def new_sample(self, delta_velocity, dt):
        """Feed one integrated accel fragment while collecting.

        Returns True once the current orientation's window is complete.
        """
        if self._status != CAL_COLLECTING_SAMPLE:
            return False
        if self._samples_collected >= self._num_samples:
            self._status = CAL_FAILED
            return False

        i = self._samples_collected
        self._dvel[i] += np.asarray(delta_velocity, dtype=float)
        self._dt[i] += dt

        if self._dt[i] <= self._sample_time:
            return False

        sample = self._dvel[i] / self._dt[i]
        if not self._accept_sample(sample):
            self._status = CAL_FAILED
            return True

        self._samples_collected += 1
        if self._samples_collected >= self._num_samples:
            self._fitness = self._run_fit(MAX_ITERATIONS)
            if self._fitness < self._tolerance and self._accept_result():
                self._status = CAL_SUCCESS
            else:
                self._status = CAL_FAILED
        else:
            self._status = CAL_WAITING_FOR_ORIENTATION
        return True

    def _get_sample(self, i):
        return self._dvel[i] / self._dt[i]

    def _accept_sample(self, sample):
        for i in range(self._samples_collected):
            if np.linalg.norm(self._get_sample(i) - sample) < self._min_sample_dist:
                return False
        return True

    def _accept_result(self):
        off = self._param[0:3]
        diag = self._param[3:6]
        if (np.any(np.abs(off) > GRAVITY_MSS) or
                np.any(diag < 0.8) or np.any(diag > 1.2)):
            return False
        return True

    # ----- Gauss-Newton fit -------------------------------------------
    def _num_params(self):
        return 6 if self._fit_type == FIT_AXIS_ALIGNED else 9

    def _calc_residual(self, sample, p):
        M = _build_M(p)
        return GRAVITY_MSS - np.linalg.norm(M @ (sample + p[0:3]))

    def _calc_mean_squared_residuals(self, p):
        if self._samples_collected == 0:
            return 1.0e30
        s = 0.0
        for i in range(self._samples_collected):
            r = self._calc_residual(self._get_sample(i), p)
            s += r * r
        return s / self._samples_collected

    def _calc_jacob(self, sample, p):
        """Partial derivatives of the residual wrt each of the 9 params."""
        offset = p[0:3]
        dx, dy, dz = p[3], p[4], p[5]
        odx, ody, odz = p[6], p[7], p[8]
        M = _build_M(p)

        sx, sy, sz = (sample + offset)
        A = dx * sx + odx * sy + ody * sz
        B = odx * sx + dy * sy + odz * sz
        C = ody * sx + odz * sy + dz * sz
        length = np.linalg.norm(M @ (sample + offset))

        ret = np.zeros(9)
        # 0-2: offsets
        ret[0] = -((dx * A) + (odx * B) + (ody * C)) / length
        ret[1] = -((odx * A) + (dy * B) + (odz * C)) / length
        ret[2] = -((ody * A) + (odz * B) + (dz * C)) / length
        # 3-5: diagonals (scale)
        ret[3] = -(sx * A) / length
        ret[4] = -(sy * B) / length
        ret[5] = -(sz * C) / length
        # 6-8: off-diagonals (cross-axis)
        ret[6] = -((sy * A) + (sx * B)) / length
        ret[7] = -((sz * A) + (sx * C)) / length
        ret[8] = -((sz * B) + (sy * C)) / length
        return ret

    def _run_fit(self, max_iterations):
        n = self._num_params()
        fitness = self._calc_mean_squared_residuals(self._param)
        min_fitness = fitness
        fit_param = self._param.copy()

        for _ in range(max_iterations):
            JTJ = np.zeros((n, n))
            JTFI = np.zeros(n)
            for k in range(self._samples_collected):
                sample = self._get_sample(k)
                jac = self._calc_jacob(sample, fit_param)
                res = self._calc_residual(sample, fit_param)
                for i in range(n):
                    JTJ[i] += jac[i] * jac[0:n]
                    JTFI[i] += jac[i] * res
            try:
                JTJ_inv = np.linalg.inv(JTJ)
            except np.linalg.LinAlgError:
                return fitness
            fit_param[0:n] -= JTJ_inv @ JTFI

            fitness = self._calc_mean_squared_residuals(fit_param)
            if not np.isfinite(fitness):
                return min_fitness
            if fitness < min_fitness:
                min_fitness = fitness
                self._param = fit_param.copy()
        return min_fitness

    # ----- results -----------------------------------------------------
    def get_calibration(self):
        """Return (offset, diag, offdiag) in firmware convention.

        Note the firmware negates the fitted offset before reporting.
        """
        offset = -self._param[0:3]
        diag = self._param[3:6].copy()
        offdiag = self._param[6:9].copy()
        return offset, diag, offdiag


# ----------------------------------------------------------------------
# Text UI driving the calibrator
# ----------------------------------------------------------------------
class CalibrationUI:
    def __init__(self, source, fit_type, num_samples, sample_time):
        self.source = source            # provides .get_accel() -> (vec3, dt) or None
        self.cal = AccelCalibrator()
        self.fit_type = fit_type
        self.num_samples = num_samples
        self.sample_time = sample_time

    def run(self):
        self.cal.start(self.fit_type, self.num_samples, self.sample_time)
        if self.cal.status == CAL_FAILED:
            print("Failed to start: too few samples for the chosen fit type.")
            return 1

        print("=" * 60)
        print(" ArduPilot-style accelerometer calibration (Gauss-Newton)")
        print("=" * 60)
        print(f" Fit type : {'ellipsoid' if self.fit_type else 'axis-aligned'}")
        print(f" Positions: {self.num_samples}")
        print(f" Avg time : {self.sample_time:.1f} s per position")
        print("-" * 60)

        for idx in range(self.num_samples):
            prompt = ORIENTATION_PROMPTS[min(idx, len(ORIENTATION_PROMPTS) - 1)]
            try:
                input(f"\n[{idx + 1}/{self.num_samples}] Place vehicle {prompt}, "
                      "hold still and press ENTER... ")
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return 1

            self.cal.collect_sample()
            self._collect_one_position(idx)

            if self.cal.status == CAL_FAILED:
                print("\n  ** Sample rejected (too close to a previous "
                      "orientation, or fit failed). Calibration FAILED. **")
                return 1

        return self._report()

    def _collect_one_position(self, idx):
        """Pump samples from the source until the averaging window fills."""
        start = time.time()
        last_print = 0.0
        while True:
            item = self.source.get_accel()
            if item is None:
                # no data yet; guard against a stalled stream
                if time.time() - start > self.sample_time * 4 + 2.0:
                    print("\n  ** Timed out waiting for IMU data. **")
                    self.cal._status = CAL_FAILED
                    return
                time.sleep(0.005)
                continue
            accel, dt = item
            done = self.cal.new_sample(np.asarray(accel) * dt, dt)

            now = time.time()
            if now - last_print > 0.1:
                got = self.cal._dt[idx] if idx < len(self.cal._dt) else 0.0
                frac = min(got / self.sample_time, 1.0)
                bar = "#" * int(frac * 20)
                sys.stdout.write(f"\r  collecting [{bar:<20}] {frac*100:5.1f}%")
                sys.stdout.flush()
                last_print = now

            if done or self.cal.status != CAL_COLLECTING_SAMPLE:
                sys.stdout.write("\r  collecting [####################] 100.0%\n")
                sys.stdout.flush()
                return

    def _report(self):
        if self.cal.status != CAL_SUCCESS:
            print(f"\nCalibration FAILED (fitness={self.cal.fitness:.4f}, "
                  f"tolerance={ACCEL_CAL_TOLERANCE}).")
            return 1
        offset, diag, offdiag = self.cal.get_calibration()
        print("\n" + "=" * 60)
        print(" CALIBRATION SUCCESS")
        print("=" * 60)
        print(f" fitness (mean sq residual): {self.cal.fitness:.6f}")
        print(f" INS_ACCOFFS  (offset) x/y/z: "
              f"{offset[0]:+.5f} {offset[1]:+.5f} {offset[2]:+.5f}  m/s/s")
        print(f" INS_ACCSCAL  (diag)   x/y/z: "
              f"{diag[0]:.5f} {diag[1]:.5f} {diag[2]:.5f}")
        if self.fit_type == FIT_ELLIPSOID:
            print(f" off-diagonal (cross)  x/y/z: "
                  f"{offdiag[0]:+.6f} {offdiag[1]:+.6f} {offdiag[2]:+.6f}")
        print("=" * 60)
        return 0


# ----------------------------------------------------------------------
# Data sources
# ----------------------------------------------------------------------
class ROS2ImuSource:
    """Subscribes to sensor_msgs/msg/Imu and queues linear acceleration."""

    def __init__(self, topic):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Imu

        self._rclpy = rclpy
        rclpy.init()
        self._node = Node("accel_calibrate_ros2")
        self._lock = threading.Lock()
        self._latest = None          # (accel tuple, stamp seconds)
        self._last_stamp = None

        def cb(msg):
            la = msg.linear_acceleration
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self._lock:
                self._latest = ((la.x, la.y, la.z), stamp)

        self._node.create_subscription(Imu, topic, cb, 50)
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_thread.start()
        print(f"Subscribed to '{topic}' (sensor_msgs/msg/Imu). "
              "Waiting for data...")

    def get_accel(self):
        with self._lock:
            latest = self._latest
            self._latest = None
        if latest is None:
            return None
        accel, stamp = latest
        if self._last_stamp is None or stamp <= self._last_stamp:
            dt = 0.0025  # fall back to ~400 Hz on first/duplicate stamp
        else:
            dt = stamp - self._last_stamp
        self._last_stamp = stamp
        # clamp implausible gaps so a stalled stream doesn't skew the average
        dt = float(np.clip(dt, 1e-4, 0.05))
        return accel, dt

    def shutdown(self):
        # Stop the executor first so the spin thread returns cleanly, then
        # join it before tearing down the node (avoids a teardown race).
        try:
            if self._rclpy.ok():
                self._rclpy.shutdown()
            self._spin_thread.join(timeout=2.0)
            self._node.destroy_node()
        except Exception:
            pass


class SimImuSource:
    """Synthetic source for offline testing of the fit.

    Produces noisy gravity vectors that step through the six face-down
    orientations as the operator advances, with a known truth offset/scale so
    the recovered calibration can be sanity-checked.
    """

    TRUTH_OFFSET = np.array([0.20, -0.15, 0.30])   # m/s/s
    TRUTH_SCALE = np.array([1.03, 0.98, 1.01])

    ORIENTATIONS = [
        np.array([0, 0, -1.0]),   # level
        np.array([0, 1.0, 0]),    # left
        np.array([0, -1.0, 0]),   # right
        np.array([-1.0, 0, 0]),   # nose down
        np.array([1.0, 0, 0]),    # nose up
        np.array([0, 0, 1.0]),    # back
        np.array([1, 1, 1]) / np.sqrt(3),
        np.array([-1, 1, 1]) / np.sqrt(3),
        np.array([1, -1, 1]) / np.sqrt(3),
        np.array([1, 1, -1]) / np.sqrt(3),
    ]

    def __init__(self, num_samples):
        self._idx = 0
        self._num = num_samples
        self._rng = np.random.default_rng(1)

    def advance(self):
        self._idx = min(self._idx + 1, len(self.ORIENTATIONS) - 1)

    def get_accel(self):
        truth = self.ORIENTATIONS[self._idx] * GRAVITY_MSS
        # apply inverse of the correction the calibrator should recover:
        # measured = truth/scale - offset  (so that scale*(meas+offset)=truth)
        measured = truth / self.TRUTH_SCALE - self.TRUTH_OFFSET
        measured = measured + self._rng.normal(0, 0.02, 3)
        return measured, 0.0025


class SimulatedUI(CalibrationUI):
    """Non-interactive variant that auto-advances the simulated source."""

    def run(self):
        self.cal.start(self.fit_type, self.num_samples, self.sample_time)
        print("Running offline simulation (no ROS 2)...")
        print(f"  truth offset = {SimImuSource.TRUTH_OFFSET}")
        print(f"  truth scale  = {SimImuSource.TRUTH_SCALE}\n")
        for idx in range(self.num_samples):
            prompt = ORIENTATION_PROMPTS[min(idx, len(ORIENTATION_PROMPTS) - 1)]
            print(f"[{idx + 1}/{self.num_samples}] position: {prompt}")
            self.source._idx = idx
            self.cal.collect_sample()
            self._collect_one_position(idx)
            if self.cal.status == CAL_FAILED:
                print("  ** FAILED **")
                return 1
        return self._report()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default="/imu",
                    help="ROS 2 sensor_msgs/msg/Imu topic (default: /imu)")
    ap.add_argument("--fit", choices=["axis", "ellipsoid"], default="axis",
                    help="fit type (default: axis)")
    ap.add_argument("--samples", type=int, default=None,
                    help="number of orientations (default: 6 for axis, 8 for ellipsoid)")
    ap.add_argument("--sample-time", type=float, default=1.0,
                    help="averaging window per position, seconds (default: 1.0)")
    ap.add_argument("--simulate", action="store_true",
                    help="run an offline self-test without ROS 2")
    args = ap.parse_args()

    fit_type = FIT_ELLIPSOID if args.fit == "ellipsoid" else FIT_AXIS_ALIGNED
    if args.samples is None:
        num_samples = 8 if fit_type == FIT_ELLIPSOID else 6
    else:
        num_samples = args.samples

    if args.simulate:
        source = SimImuSource(num_samples)
        ui = SimulatedUI(source, fit_type, num_samples, args.sample_time)
        return ui.run()

    try:
        source = ROS2ImuSource(args.topic)
    except ImportError:
        print("ERROR: ROS 2 (rclpy / sensor_msgs) not available. "
              "Source a ROS 2 workspace, or use --simulate.", file=sys.stderr)
        return 2

    ui = CalibrationUI(source, fit_type, num_samples, args.sample_time)
    try:
        return ui.run()
    finally:
        source.shutdown()


if __name__ == "__main__":
    sys.exit(main())
