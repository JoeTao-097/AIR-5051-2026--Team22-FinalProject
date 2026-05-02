#!/usr/bin/env python3
"""Scan execution node.

Uses MoveIt's Cartesian path planning to follow scan paths generated
by scan_planner_node. Works entirely in ExternalControl mode.

For force control, URScript can be sent via the secondary interface
after switching to Remote Control mode on the teach pendant.
"""

import os
import subprocess
import threading
import numpy as np
import yaml
import rospy
import rospkg
import actionlib
import moveit_commander
import tf2_ros
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, WrenchStamped
from moveit_msgs.msg import DisplayTrajectory
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
import transforms3d.quaternions as tq


class ForceScanNode:
    def __init__(self):
        rospy.init_node('force_scan_node')

        self.robot_ip = rospy.get_param('/robot/ip', '192.168.1.10')
        self.scan_speed = rospy.get_param('/scan/scan_speed', 0.01)
        self.approach_height = rospy.get_param('/scan/approach_height', 0.05)

        # ── Acquisition quality controls ─────────────────────────────
        self.auto_record = rospy.get_param('/scan/auto_record', True)
        self.stable_wait_before = rospy.get_param('/scan/stable_wait_before', 0.7)
        self.stable_wait_after = rospy.get_param('/scan/stable_wait_after', 0.3)
        self.zero_ft_sensor = rospy.get_param('/scan/zero_ft_sensor', True)
        self.hide_cursor_flag = rospy.get_param('/scan/hide_cursor', True)

        # ── Force-adaptive scan (per-segment Fz monitoring) ──────────
        self.force_adaptive_scan = rospy.get_param(
            '/scan/force_adaptive_scan', True)
        self.force_adaptive_min_force = rospy.get_param(
            '/scan/force_adaptive_min_force', 2.0)
        self.force_adaptive_step = rospy.get_param(
            '/scan/force_adaptive_step', 0.001)
        self.force_adaptive_max_push = rospy.get_param(
            '/scan/force_adaptive_max_push', 0.010)
        self.force_adaptive_settle_s = rospy.get_param(
            '/scan/force_adaptive_settle_s', 0.15)

        # ── Touchdown Z-calibration (force-based) ────────────────────
        self.touchdown_enabled = rospy.get_param('/scan/touchdown_enabled', True)
        self.touchdown_force = rospy.get_param('/scan/touchdown_force', 2.5)
        self.touchdown_max_descent = rospy.get_param('/scan/touchdown_max_descent', 0.10)
        self.touchdown_speed = rospy.get_param('/scan/touchdown_speed', 0.002)
        self.touchdown_extra_press = rospy.get_param('/scan/touchdown_extra_press', 0.0)
        # Two-phase descent: fast cruise down to ~`pre_contact_buffer`
        # above the planned-surface Z, then slow approach for the final
        # `pre_contact_buffer` (where actual contact may occur).
        self.touchdown_fast_speed = rospy.get_param('/scan/touchdown_fast_speed', 0.020)
        self.touchdown_pre_contact_buffer = rospy.get_param(
            '/scan/touchdown_pre_contact_buffer', 0.030)

        # Live wrench tracking (for touchdown contact detection)
        self._wrench_lock = threading.Lock()
        self._wrench_z = 0.0
        self._wrench_z_baseline = 0.0
        self._wrench_received = False
        rospy.Subscriber('/wrench', WrenchStamped, self._wrench_cb, queue_size=1)

        self.scan_path = None
        self.scanning = False

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        rospack = rospkg.RosPack()
        self._home_yaml_path = os.path.join(
            rospack.get_path('us3d_bringup'), 'config', 'home_position.yaml')

        self._joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self._current_joints = None
        rospy.Subscriber('/joint_states', JointState, self._joint_states_cb)

        self._traj_client = actionlib.SimpleActionClient(
            '/scaled_pos_joint_traj_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)

        moveit_commander.roscpp_initialize([])
        self._group = moveit_commander.MoveGroupCommander("manipulator")
        self._group.set_max_velocity_scaling_factor(0.1)
        self._group.set_max_acceleration_scaling_factor(0.1)
        # RRTConnect: returns immediately when first solution is found
        # (vs RRTstar which over-optimises for the full planning_time
        # budget and routinely produces detours through joint space
        # — observed cost 12.9 rad for a 40° wrist rotation request).
        # Path quality from RRTConnect is fine for free-space approach
        # moves and avoids the wasted 10 seconds + the wrist-loop detours.
        try:
            self._group.set_planner_id('RRTConnectkConfigDefault')
        except Exception:
            try:
                self._group.set_planner_id('RRTConnect')
            except Exception:
                pass
        self._group.set_planning_time(2.5)            # was 10.0 (90% wasted)
        self._group.set_num_planning_attempts(3)
        # Allow MoveIt to cull goal candidates that require >180° wrist
        # roll-around. (Set joint goal tolerance loose so the IK solver
        # picks the closest-to-current solution.)
        try:
            self._group.set_goal_orientation_tolerance(0.01)  # ~0.6°
            self._group.set_goal_position_tolerance(0.001)    # 1 mm
        except Exception:
            pass

        self.pub_display_traj = rospy.Publisher(
            '/move_group/display_planned_path', DisplayTrajectory, queue_size=1)
        self._preview_plan = None

        rospy.Subscriber('/us3d/scan_path', PoseArray, self._path_cb)

        rospy.Service('/us3d/preview_scan', Trigger, self._preview_scan_srv)
        rospy.Service('/us3d/start_scan', Trigger, self._start_scan_srv)
        rospy.Service('/us3d/stop_scan', Trigger, self._stop_scan_srv)
        rospy.Service('/us3d/touch_marker', Trigger, self._touch_marker_srv)
        rospy.Service('/us3d/record_home', Trigger, self._record_home_srv)
        rospy.Service('/us3d/move_home', Trigger, self._move_home_srv)
        # Manual escape hatch: if anything goes wrong and the speed
        # slider stays low, user can call this to force it to 1.0
        rospy.Service('/us3d/restore_speed', Trigger,
                      lambda req: self._restore_speed_srv())

        # Safety: always reset speed slider to 1.0 on shutdown so
        # subsequent sessions don't inherit a slow slider.
        rospy.on_shutdown(lambda: self._restore_speed_slider(fraction=1.0))

        self._markers = None
        rospy.Subscriber('/us3d/markers', PoseArray, self._markers_cb)

        # Optional service handles for auto-record (lazily resolved)
        self._record_start_proxy = None
        self._record_stop_proxy = None

        rospy.loginfo(
            "Force scan node ready (MoveIt Cartesian, speed=%.1f mm/s, "
            "auto_record=%s, stable_wait=%.1f/%.1fs)",
            self.scan_speed * 1000, self.auto_record,
            self.stable_wait_before, self.stable_wait_after)

        rospy.Timer(rospy.Duration(3.0), self._startup_move_home, oneshot=True)

    # ── Wrench monitoring ───────────────────────────────────────

    def _wrench_cb(self, msg):
        """Cache the latest Fz reading for touchdown contact detection.

        Note: /wrench from ur_robot_driver is published in the
        base_link frame by default. Fz here is the vertical force; a
        downward reaction force from the surface gives Fz > 0 (pushing
        the TCP up). Polarity may differ depending on calibration; we
        compare to a baseline taken just before descent.
        """
        with self._wrench_lock:
            self._wrench_z = msg.wrench.force.z
            self._wrench_received = True

    def _get_wrench_z(self):
        with self._wrench_lock:
            return self._wrench_z

    def _wait_for_wrench(self, timeout=2.0):
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            with self._wrench_lock:
                if self._wrench_received:
                    return True
            if (rospy.Time.now() - t0).to_sec() > timeout:
                return False
            rospy.sleep(0.05)

    # ── Octomap helpers ──────────────────────────────────────────

    def _clear_octomap(self):
        """Wipe the planning-scene octomap so depth-cloud noise that's
        in front of the eye-in-hand camera (or stale voxels next to
        the wrist) doesn't block approach planning. Octomap will start
        re-populating immediately from the next depth frame, but having
        it briefly empty buys us a clean planning + execution window."""
        try:
            from std_srvs.srv import Empty
            srv = rospy.ServiceProxy('/clear_octomap', Empty)
            srv.wait_for_service(timeout=1.0)
            srv()
            rospy.loginfo("Octomap cleared")
            return True
        except Exception as e:
            rospy.logdebug("clear_octomap failed: %s", e)
            return False

    def _move_to_pose(self, target_pose, vel_scale=0.1, allow_cartesian=True):
        """Move TCP to target pose with two-stage strategy:

        1. **Try Cartesian SLERP first** (straight-line + smooth wrist
           rotation, no joint-space detour). This usually succeeds for
           approach/retract since they are short translations + a
           bounded wrist rotation. Cartesian solutions are by definition
           the SHORTEST tool-path possible.
        2. **Fall back to RRTConnect** if Cartesian path is < 95%
           feasible (e.g., the segment crosses a singularity or
           requires a non-monotonic IK switch).

        Skipping straight to RRTConnect was producing absurd detours
        (observed 12.9 rad joint-space path for a 40° wrist request).
        """
        # Always clear octomap before a free-space motion so the
        # planning scene matches the snapshot used for planning.
        self._clear_octomap()
        rospy.sleep(0.1)

        # Sync planning start state with reality (avoids
        # "start point deviates from current robot state" errors).
        self._reset_start_state()
        self._group.set_max_velocity_scaling_factor(vel_scale)

        # ── Attempt 1: Cartesian SLERP (fast, deterministic, shortest path)
        if allow_cartesian:
            try:
                plan, fraction = self._group.compute_cartesian_path(
                    [target_pose], 0.005, 0.0, avoid_collisions=False)
            except TypeError:
                plan, fraction = self._group.compute_cartesian_path(
                    [target_pose], 0.005, 0.0, False)
            if fraction >= 0.95:
                rospy.loginfo("Move: Cartesian SLERP (%d points, %.0f%% coverage)",
                              len(plan.joint_trajectory.points), fraction * 100)
                success = self._group.execute(plan, wait=True)
                self._group.stop()
                if success:
                    return True
                rospy.logwarn("Cartesian execution failed — falling back to RRT")
            else:
                rospy.loginfo("Cartesian SLERP only %.0f%% feasible "
                              "(probably IK switch / singularity) — "
                              "falling back to RRTConnect", fraction * 100)

        # ── Attempt 2: RRTConnect (tolerant of singularities, may detour)
        self._reset_start_state()
        self._group.set_pose_target(target_pose)
        success = self._group.go(wait=True)
        self._group.stop()
        self._group.clear_pose_targets()
        return bool(success)

    # ── UR speed slider control ──────────────────────────────────

    def _set_speed_slider_for_scan(self):
        """Set the UR speed slider (teach pendant equivalent) so the
        scan executes at the configured scan_speed.

        The UR speed slider scales ALL robot motion velocities by a
        fraction in [0.01, 1.0]. This is a global runtime override
        that's much more reliable than MoveIt's velocity_scaling_factor
        on Cartesian paths.

        Empirical: at slider=1.0, MoveIt Cartesian scan runs at
        roughly the natural joint velocity = ~15 mm/s for a typical
        scan path. So slider = scan_speed / 15mm/s gives the right
        scaling. We err on the safe side using 20 mm/s as the
        reference.

        Returns the previous slider value so we can restore it,
        or None if setting failed.
        """
        try:
            from ur_msgs.srv import SetSpeedSliderFraction
            srv = rospy.ServiceProxy(
                '/ur_hardware_interface/set_speed_slider',
                SetSpeedSliderFraction)
            srv.wait_for_service(timeout=2.0)
        except Exception as e:
            rospy.logwarn("UR speed slider service unavailable (%s); "
                          "scan will run at default fast speed", e)
            return None

        # Save current slider value (assume 1.0 if unknown — UR driver
        # doesn't expose a getter; we'll restore to 1.0 after)
        prev_slider = 1.0

        # Reference natural Cartesian speed at slider=1.0 (empirical)
        ref_speed_at_full_slider = rospy.get_param(
            '/scan/speed_slider_reference_mms', 20.0) / 1000.0   # m/s

        target_fraction = self.scan_speed / ref_speed_at_full_slider
        target_fraction = max(0.01, min(1.0, target_fraction))

        try:
            resp = srv(target_fraction)
            ok = getattr(resp, 'success', True)
            rospy.loginfo(
                "Speed slider set to %.3f (%.0f%%) for scan at %.1f mm/s "
                "(target / %.1f mm/s ref): success=%s",
                target_fraction, target_fraction * 100,
                self.scan_speed * 1000,
                ref_speed_at_full_slider * 1000, ok)
            if ok:
                self._prev_speed_slider = prev_slider
                return prev_slider
        except Exception as e:
            rospy.logwarn("Failed to set speed slider: %s", e)
        return None

    def _restore_speed_srv(self):
        """ROS service: manually restore UR speed slider to 1.0.
        Use after a scan was interrupted and slider got stuck low."""
        try:
            self._restore_speed_slider(fraction=1.0)
            return TriggerResponse(success=True,
                                   message="Speed slider restored to 1.0")
        except Exception as e:
            return TriggerResponse(success=False,
                                   message="Failed: %s" % e)

    def _restore_speed_slider(self, fraction=None):
        """Restore the UR speed slider to the value before scan
        (or to a specific fraction)."""
        try:
            from ur_msgs.srv import SetSpeedSliderFraction
            srv = rospy.ServiceProxy(
                '/ur_hardware_interface/set_speed_slider',
                SetSpeedSliderFraction)
            srv.wait_for_service(timeout=2.0)
            if fraction is None:
                fraction = getattr(self, '_prev_speed_slider', 1.0)
            resp = srv(fraction)
            rospy.loginfo("Speed slider restored to %.3f", fraction)
        except Exception as e:
            rospy.logwarn("Failed to restore speed slider: %s", e)

    # ── Acquisition helpers ──────────────────────────────────────

    def _hide_cursor(self):
        """Move the mouse pointer off-screen so it does not appear in the
        US capture frame. Best-effort — silently ignores failure."""
        if not self.hide_cursor_flag:
            return
        try:
            subprocess.run(
                ['xdotool', 'mousemove', '4000', '4000'],
                check=False, timeout=2.0,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            rospy.logdebug("hide_cursor failed: %s", e)

    def _zero_ft_sensor(self):
        """Reset F/T sensor bias while suspended above the surface.

        Uses RTDE if available; falls back to UR's standard ROS service
        (/ur_hardware_interface/zero_ftsensor) otherwise.
        """
        if not self.zero_ft_sensor:
            return
        # Try ROS service first (no extra deps)
        try:
            srv = rospy.ServiceProxy(
                '/ur_hardware_interface/zero_ftsensor', Trigger)
            srv.wait_for_service(timeout=1.0)
            resp = srv()
            if resp.success:
                rospy.loginfo("F/T sensor zeroed via ROS service")
                return
        except Exception:
            pass
        # Fall back to RTDE
        try:
            import rtde_control
            rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            ok = rtde_c.zeroFtSensor()
            rtde_c.disconnect()
            rospy.loginfo("F/T sensor zeroed via RTDE: %s", ok)
        except Exception as e:
            rospy.logwarn("F/T zero failed: %s", e)

    def _start_recording(self):
        if not self.auto_record:
            return
        try:
            if self._record_start_proxy is None:
                rospy.wait_for_service('/us3d/start_recording', timeout=2.0)
                self._record_start_proxy = rospy.ServiceProxy(
                    '/us3d/start_recording', Trigger)
            resp = self._record_start_proxy(TriggerRequest())
            rospy.loginfo("auto-record start: %s", resp.message)
        except Exception as e:
            rospy.logwarn("auto-record start failed: %s", e)

    def _stop_recording(self):
        if not self.auto_record:
            return
        try:
            if self._record_stop_proxy is None:
                rospy.wait_for_service('/us3d/stop_recording', timeout=2.0)
                self._record_stop_proxy = rospy.ServiceProxy(
                    '/us3d/stop_recording', Trigger)
            resp = self._record_stop_proxy(TriggerRequest())
            rospy.loginfo("auto-record stop: %s", resp.message)
        except Exception as e:
            rospy.logwarn("auto-record stop failed: %s", e)

    # ── Joint state ──────────────────────────────────────────────

    def _joint_states_cb(self, msg):
        try:
            positions = []
            for name in self._joint_names:
                idx = msg.name.index(name)
                positions.append(msg.position[idx])
            self._current_joints = positions
        except ValueError:
            pass

    # ── Home position ────────────────────────────────────────────

    def _startup_move_home(self, event):
        msg = self._try_move_home()
        rospy.loginfo("Startup home move: %s", msg)

    def _load_home_config(self):
        enabled = rospy.get_param('/home_position/enabled', False)
        if not enabled:
            return None
        joints = rospy.get_param('/home_position/joints', None)
        if joints is None or len(joints) != 6:
            return None
        duration = rospy.get_param('/home_position/move_duration', 5.0)
        return {'joints': joints, 'duration': duration}

    def _try_move_home(self):
        home = self._load_home_config()
        if home is None:
            return "Home position not configured, skipped."
        if not self._traj_client.wait_for_server(rospy.Duration(10.0)):
            return "Trajectory action server not available."

        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self._joint_names
        point = JointTrajectoryPoint()
        point.positions = home['joints']
        point.velocities = [0.0] * 6
        point.time_from_start = rospy.Duration(home['duration'])
        goal.trajectory.points.append(point)

        rospy.loginfo("Moving to home position: %s", home['joints'])
        self._traj_client.send_goal(goal)
        self._traj_client.wait_for_result(rospy.Duration(home['duration'] + 5.0))

        result = self._traj_client.get_result()
        if result and result.error_code == 0:
            rospy.loginfo("Reached home position.")
            return "Moved to home position."
        msg = "Failed to move to home (error_code=%s)" % (
            result.error_code if result else "timeout")
        rospy.logwarn(msg)
        return msg

    def _record_home_srv(self, req):
        if self._current_joints is None:
            return TriggerResponse(success=False, message="No joint states received yet")
        try:
            joints_rounded = [round(j, 6) for j in self._current_joints]
            home_data = {
                'home_position': {
                    'enabled': True,
                    'joints': joints_rounded,
                    'move_duration': rospy.get_param('/home_position/move_duration', 5.0),
                }
            }
            with open(self._home_yaml_path, 'w') as f:
                f.write("# Home/start position for the robot arm (joint angles in radians)\n")
                f.write("# Auto-generated by /us3d/record_home service.\n")
                f.write("# The robot will move to this position after connecting.\n")
                yaml.dump(home_data, f, default_flow_style=False)
            rospy.set_param('/home_position/enabled', True)
            rospy.set_param('/home_position/joints', joints_rounded)
            msg = "Home position saved: %s" % joints_rounded
            rospy.loginfo(msg)
            return TriggerResponse(success=True, message=msg)
        except Exception as e:
            return TriggerResponse(success=False, message="Record failed: %s" % str(e))

    def _move_home_srv(self, req):
        msg = self._try_move_home()
        success = "Moved to home" in msg
        return TriggerResponse(success=success, message=msg)

    # ── Markers ───────────────────────────────────────────────────

    def _markers_cb(self, msg):
        self._markers = msg

    # ── Touch marker test ────────────────────────────────────────

    def _touch_marker_srv(self, req):
        """Move probe tip to marker center with full diagnostic output."""
        if self._markers is None or len(self._markers.poses) < 1:
            return TriggerResponse(success=False,
                                   message="No markers detected. Call /us3d/detect_markers first.")

        marker_pose = self._markers.poses[-1]
        mx = marker_pose.position.x
        my = marker_pose.position.y
        mz = marker_pose.position.z

        current_pose = self._group.get_current_pose().pose

        # TF diagnostic: check tool0 and probe_tip positions independently
        try:
            tf_tool0 = self._tf_buffer.lookup_transform(
                'base_link', 'tool0', rospy.Time(0), rospy.Duration(1.0))
            tf_probe = self._tf_buffer.lookup_transform(
                'base_link', 'probe_tip', rospy.Time(0), rospy.Duration(1.0))
            tool0_z = tf_tool0.transform.translation.z
            probe_z = tf_probe.transform.translation.z
            probe_z_offset = tool0_z - probe_z
            rospy.loginfo("=== DIAGNOSTIC ===")
            rospy.loginfo("  TF tool0  in base_link: z=%.4f", tool0_z)
            rospy.loginfo("  TF probe_tip in base_link: z=%.4f", probe_z)
            rospy.loginfo("  Probe Z offset (tool0-probe_tip): %.4f m", probe_z_offset)
            rospy.loginfo("  MoveIt current_pose.z: %.4f", current_pose.position.z)
            rospy.loginfo("  Marker position: (%.4f, %.4f, %.4f)", mx, my, mz)
            rospy.loginfo("  Target tool0.z = marker.z + offset = %.4f + %.4f = %.4f",
                          mz, probe_z_offset, mz + probe_z_offset)
            rospy.loginfo("  Expected probe_tip.z after move: %.4f (should ≈ marker.z)", mz)
        except Exception as e:
            rospy.logwarn("TF diagnostic failed: %s", e)
            probe_z_offset = rospy.get_param('/scan/probe_length', 0.160)
            rospy.loginfo("Using fallback probe_length: %.4f m", probe_z_offset)

        # Step 1: Move above marker
        above_pose = Pose()
        above_pose.position.x = mx
        above_pose.position.y = my
        above_pose.position.z = mz + probe_z_offset + self.approach_height
        above_pose.orientation = current_pose.orientation

        rospy.loginfo("Step 1: Moving above marker at tool0.z=%.4f ...",
                      above_pose.position.z)
        self._group.set_max_velocity_scaling_factor(0.1)
        self._group.set_pose_target(above_pose)
        success = self._group.go(wait=True)
        self._group.stop()
        self._group.clear_pose_targets()

        if not success:
            return TriggerResponse(success=False,
                                   message="Failed to move above marker.")

        # Verify approach position reached
        after_approach = self._group.get_current_pose().pose
        rospy.loginfo("  After approach: tool0.z=%.4f (target was %.4f, diff=%.1fmm)",
                      after_approach.position.z, above_pose.position.z,
                      (after_approach.position.z - above_pose.position.z) * 1000)

        # Step 2: Descend to marker
        touch_pose = Pose()
        touch_pose.position.x = mx
        touch_pose.position.y = my
        touch_pose.position.z = mz + probe_z_offset
        touch_pose.orientation = current_pose.orientation

        rospy.loginfo("Step 2: Descending %.1fmm to tool0.z=%.4f ...",
                      self.approach_height * 1000, touch_pose.position.z)

        plan, fraction = self._group.compute_cartesian_path(
            [touch_pose], 0.001, 0.0)

        rospy.loginfo("  Cartesian plan fraction: %.1f%%", fraction * 100)

        if fraction < 0.9:
            rospy.logwarn("  Cartesian descent only %.0f%% — trying joint-space move",
                          fraction * 100)
            self._group.set_max_velocity_scaling_factor(0.02)
            self._group.set_pose_target(touch_pose)
            success = self._group.go(wait=True)
            self._group.stop()
            self._group.clear_pose_targets()
            if not success:
                return TriggerResponse(
                    success=False,
                    message="Descent failed (Cartesian %.0f%%, joint-space also failed)" %
                            (fraction * 100))
        else:
            self._group.set_max_velocity_scaling_factor(0.02)
            self._group.execute(plan, wait=True)
            self._group.stop()

        # Step 3: Verify final position
        final_pose = self._group.get_current_pose().pose
        try:
            tf_final = self._tf_buffer.lookup_transform(
                'base_link', 'probe_tip', rospy.Time(0), rospy.Duration(1.0))
            actual_probe_z = tf_final.transform.translation.z
            error_mm = (actual_probe_z - mz) * 1000
            rospy.loginfo("=== RESULT ===")
            rospy.loginfo("  Final tool0.z:      %.4f (target: %.4f)",
                          final_pose.position.z, touch_pose.position.z)
            rospy.loginfo("  Final probe_tip.z:  %.4f (marker.z: %.4f)",
                          actual_probe_z, mz)
            rospy.loginfo("  Z error: %.1f mm (%s)",
                          abs(error_mm),
                          "probe too HIGH" if error_mm > 0 else "probe too LOW")
            msg = ("probe_tip.z=%.4f, marker.z=%.4f, error=%.1fmm (%s)" %
                   (actual_probe_z, mz, error_mm,
                    "HIGH" if error_mm > 0 else "LOW"))
        except Exception:
            msg = "Moved to marker (%.4f, %.4f, %.4f), TF verify failed" % (mx, my, mz)

        rospy.loginfo(msg)
        return TriggerResponse(success=True, message=msg)

    # ── Probe offset ─────────────────────────────────────────────

    def _get_probe_z_offset(self):
        """Compute the Z-component of tool0→probe_tip in base_link frame.

        Uses TF to account for current tool orientation — returns the
        actual vertical distance between tool0 and probe_tip rather than
        the fixed probe_length parameter (which is only correct when the
        probe points straight down).
        """
        try:
            t = self._tf_buffer.lookup_transform(
                'base_link', 'tool0', rospy.Time(0), rospy.Duration(0.5))
            t2 = self._tf_buffer.lookup_transform(
                'base_link', 'probe_tip', rospy.Time(0), rospy.Duration(0.5))
            dz = t.transform.translation.z - t2.transform.translation.z
            if dz < 0.01:
                rospy.logwarn("probe_tip is above tool0 (dz=%.4f), "
                             "falling back to probe_length param", dz)
                return rospy.get_param('/scan/probe_length', 0.160)
            return dz
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("TF lookup failed (%s), using probe_length param", e)
            return rospy.get_param('/scan/probe_length', 0.160)

    # ── Scan path ────────────────────────────────────────────────

    def _path_cb(self, msg):
        self.scan_path = msg
        rospy.loginfo("Received scan path with %d waypoints", len(msg.poses))

    # ── Preview & Scan execution via MoveIt Cartesian path ──────

    def _get_probe_offset_in_tool0(self):
        """Return the (x,y,z) offset from tool0 to probe_tip expressed
        in the tool0 frame. Cached after the first TF lookup since the
        probe is rigidly attached.

        We need the FULL 3D offset (not just the base-Z projection)
        so that for any probe orientation we can compute:

            tool0_target = probe_tip_target - R(orientation) @ offset

        which keeps probe_tip exactly at the planner waypoint XYZ
        regardless of how the wrist is rotated.
        """
        if hasattr(self, '_probe_offset_tool0'):
            return self._probe_offset_tool0
        try:
            t = self._tf_buffer.lookup_transform(
                'tool0', 'probe_tip', rospy.Time(0), rospy.Duration(1.0))
            self._probe_offset_tool0 = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z])
            rospy.loginfo("Probe offset in tool0 frame: (%.4f, %.4f, %.4f) m, "
                          "|offset|=%.4f m",
                          *self._probe_offset_tool0,
                          float(np.linalg.norm(self._probe_offset_tool0)))
            return self._probe_offset_tool0
        except Exception as e:
            rospy.logwarn("TF tool0→probe_tip failed (%s), using "
                          "fallback (0,0,probe_length)", e)
            length = rospy.get_param('/scan/probe_length', 0.160)
            self._probe_offset_tool0 = np.array([0.0, 0.0, length])
            return self._probe_offset_tool0

    def _tool0_for_probe_tip(self, tip_xyz, orientation):
        """Compute the tool0 pose target so that probe_tip lands exactly
        at ``tip_xyz`` when the wrist is at ``orientation``.

        tool0 = tip_xyz - R(orientation) @ probe_offset_in_tool0
        """
        offset_tool0 = self._get_probe_offset_in_tool0()
        R = tq.quat2mat([orientation.w, orientation.x,
                         orientation.y, orientation.z])
        offset_base = R @ offset_tool0
        return (tip_xyz[0] - offset_base[0],
                tip_xyz[1] - offset_base[1],
                tip_xyz[2] - offset_base[2])

    def _build_approach_pose(self, keypoints, orientation, probe_z_offset=None):
        """Build an approach pose ABOVE the first keypoint such that
        probe_tip will be at (wp[0].xy, wp[0].z + approach_height) when
        the wrist holds ``orientation``. Uses full 3D probe-tip
        compensation so that a tilted wrist doesn't shift probe_tip
        sideways."""
        wp0 = keypoints[0].position
        # Target probe_tip position: directly above wp[0] by approach_height
        tip_target = np.array([wp0.x, wp0.y, wp0.z + self.approach_height])
        tx, ty, tz = self._tool0_for_probe_tip(tip_target, orientation)
        p = Pose()
        p.position.x = tx
        p.position.y = ty
        p.position.z = tz
        p.orientation = orientation
        rospy.loginfo("Approach: probe_tip target=(%.3f,%.3f,%.3f) → "
                      "tool0 target=(%.3f,%.3f,%.3f)",
                      tip_target[0], tip_target[1], tip_target[2], tx, ty, tz)
        return p

    def _build_cartesian_poses(self, keypoints, orientation=None,
                               probe_z_offset=None,
                               use_planner_orientation=True,
                               dz_correction=0.0,
                               override_z=None,
                               z_profile=None,
                               min_probe_tip_z=None):
        """Build Cartesian scan poses with proper probe-tip compensation.

        For each waypoint:
            probe_tip_target.z = override_z   (if provided)
                              or wp.z + dz_correction
            probe_tip_target.xy = wp.xy
            tool0_target = probe_tip_target - R(used_orient) @ probe_offset_in_tool0

        - ``override_z``: if not None, force EVERY probe_tip target to
          this exact Z (m, base frame). Use this with the contact Z
          measured by touchdown so the entire scan stays at a known-
          good height instead of trusting the depth-camera Z curve
          (which can have 20-30mm bumps from sparse RBF artifacts).
        - ``dz_correction``: per-waypoint Z offset; ignored when
          override_z is set.
        - The R-rotated probe offset guarantees probe_tip lands
          exactly at the chosen XYZ regardless of wrist tilt.
        """
        del probe_z_offset  # unused; TF lookup gives the full 3D offset
        poses = []
        max_dq = 0.0
        prev_q = None
        for wp in keypoints:
            wp_q = wp.orientation
            wp_norm = (wp_q.x ** 2 + wp_q.y ** 2 + wp_q.z ** 2 + wp_q.w ** 2)
            if use_planner_orientation and abs(wp_norm - 1.0) < 0.05:
                used_q = wp_q
                if prev_q is not None:
                    max_dq = max(max_dq, np.degrees(
                        self._angle_between_quats(prev_q, wp_q)))
                prev_q = wp_q
            else:
                used_q = orientation

            if z_profile is not None and len(z_profile) == len(keypoints):
                # Per-waypoint Z from multi-touchdown surface measurement
                tip_z = float(z_profile[len(poses)])
            elif override_z is not None:
                tip_z = override_z
            else:
                tip_z = wp.position.z + dz_correction

            # Safety floor: never push the probe below this Z. Set to
            # the touchdown contact Z so a noisy depth-cloud Z curve
            # that says "go down" doesn't drive the probe deep into
            # the surface (which causes joint-torque protective stop
            # — UR error C157).
            if min_probe_tip_z is not None and tip_z < min_probe_tip_z:
                tip_z = min_probe_tip_z

            tip_target = np.array([wp.position.x,
                                   wp.position.y,
                                   tip_z])
            tx, ty, tz = self._tool0_for_probe_tip(tip_target, used_q)
            p = Pose()
            p.position.x = tx
            p.position.y = ty
            p.position.z = tz
            p.orientation = used_q
            poses.append(p)

        if use_planner_orientation and len(poses) > 1:
            if override_z is not None:
                rospy.loginfo(
                    "Cartesian poses: %d waypoints, max inter-waypoint "
                    "orientation jump = %.2f°, FLAT scan @ probe_tip.z="
                    "%.4f m (overrides planner Z and dz_correction)",
                    len(poses), max_dq, override_z)
            else:
                rospy.loginfo(
                    "Cartesian poses: %d waypoints, max inter-waypoint "
                    "orientation jump = %.2f°, dz_correction=%+.1fmm",
                    len(poses), max_dq, dz_correction * 1000)
        return poses

    @staticmethod
    def _quat_to_array(q):
        return np.array([q.w, q.x, q.y, q.z])

    def _pick_scan_orientation(self, keypoints):
        """Choose the orientation to which the wrist should rotate
        before descending. Prefers the planner's first-waypoint
        orientation; falls back to current pose if invalid.
        """
        if keypoints and keypoints[0].orientation is not None:
            q = keypoints[0].orientation
            n2 = q.x ** 2 + q.y ** 2 + q.z ** 2 + q.w ** 2
            if abs(n2 - 1.0) < 0.05:
                return q
        rospy.logwarn("Planner waypoint orientation invalid, "
                      "falling back to current pose orientation")
        return self._group.get_current_pose().pose.orientation

    def _angle_between_quats(self, q1, q2):
        """Return the angle (rad) between two ROS quaternions."""
        a = self._quat_to_array(q1)
        b = self._quat_to_array(q2)
        # Force same hemisphere
        if np.dot(a, b) < 0:
            b = -b
        cos_half = np.clip(abs(np.dot(a, b)), -1.0, 1.0)
        return 2.0 * np.arccos(cos_half)

    # ── Touchdown Z-calibration ──────────────────────────────────

    def _touchdown_calibrate(self, scan_orientation):
        """Slowly descend from the current pose until contact is detected.

        Returns
        -------
        dz_for_scan : float or None
            Z offset (m) to ADD to every scan waypoint so the probe
            actually touches the surface during scanning.

            Math:  dz_for_scan = real_surface_z - planned_wp_z
              =  (contact_tool0_z - probe_z_offset) - planned_wp_z

            We start descent from the APPROACH pose (= planned wp_z +
            probe_z_offset + approach_height), so the raw measurement
            ``dz_raw = contact_z - approach_z`` already includes a
            -approach_height offset; we add it back before returning.

        How it works (incremental safe descent)
        ---------------------------------------
        Instead of ONE long async Cartesian + soft-stop on contact (which
        causes ~150ms of overshoot → ~30N peak force → UR PROTECTIVE_STOP
        on rigid surfaces), we descend in MANY SHORT BLOCKING steps:

          1. Sample baseline Fz (probe in air ≈ 0)
          2. Loop:
             a. Read live Fz; if |Fz-baseline| ≥ touchdown_force → DONE
             b. Plan + execute a single 3mm Cartesian step downward
                (synchronously — the move ends with natural deceleration
                back to 0 m/s before the next iteration)
             c. Brief settle, repeat
          3. dz_correction = current_z - start_z

        Because each step ends with v=0, the ROBOT IS STATIONARY at the
        moment force is detected. No deceleration overshoot ⇒ no peak
        force spike ⇒ no PROTECTIVE_STOP.
        """
        if not self.touchdown_enabled:
            return None
        if not self._wait_for_wrench(timeout=2.0):
            rospy.logwarn("No /wrench data — skipping touchdown calibration")
            return None

        baseline = self._get_wrench_z()
        rospy.loginfo("Touchdown: baseline Fz = %+.2f N (probe suspended)",
                      baseline)
        self._wrench_z_baseline = baseline

        start_pose = self._group.get_current_pose().pose
        planned_z = start_pose.position.z

        # Use a deliberately-tiny step so micro-overshoot at end of each
        # move (always < 0.1mm with a velocity-scaling factor of 0.05) is
        # negligible compared to soft tissue compliance.
        step_m = rospy.get_param('/scan/touchdown_step', 0.003)
        n_steps = int(self.touchdown_max_descent / step_m) + 1

        # Cap the per-move acceleration AND velocity so each tiny move
        # peaks at well under 1 cm/s. Trapezoidal profile: with 3mm step
        # and 5mm/s peak, total move time ≈ 0.6-0.8s.
        speed_factor = max(0.005, min(0.05, self.touchdown_speed / 0.25))
        self._group.set_max_velocity_scaling_factor(speed_factor)
        self._group.set_max_acceleration_scaling_factor(0.05)

        rospy.loginfo(
            "Touchdown: %d × %.1fmm incremental steps "
            "(blocking), max %.0fmm, threshold |dFz|≥%.1fN",
            n_steps, step_m * 1000,
            self.touchdown_max_descent * 1000,
            self.touchdown_force)

        contact_detected = False
        descended = 0.0
        for i in range(n_steps):
            # Check force BEFORE moving — robot is currently stationary
            d_fz = abs(self._get_wrench_z() - baseline)
            if d_fz >= self.touchdown_force:
                contact_detected = True
                rospy.loginfo(
                    "Touchdown: CONTACT before step %d, "
                    "|dFz|=%.2fN, descended %.1fmm",
                    i, d_fz, descended * 1000)
                break

            current_pose = self._group.get_current_pose().pose
            target = Pose()
            target.position.x = current_pose.position.x
            target.position.y = current_pose.position.y
            target.position.z = current_pose.position.z - step_m
            target.orientation = current_pose.orientation

            self._reset_start_state()
            try:
                plan, frac = self._group.compute_cartesian_path(
                    [target], 0.001, 0.0, avoid_collisions=False)
            except TypeError:
                plan, frac = self._group.compute_cartesian_path(
                    [target], 0.001, 0.0, False)

            if frac < 0.95:
                rospy.logwarn("Touchdown: micro-step %d planning failed "
                              "(frac=%.0f%%) — stopping descent", i, frac * 100)
                break

            success = self._group.execute(plan, wait=True)
            if not success:
                rospy.logwarn("Touchdown: micro-step %d execution failed", i)
                break

            descended += step_m
            rospy.sleep(0.08)  # brief settle so wrench reading reflects current contact

            if i % 10 == 0 and i > 0:
                rospy.loginfo("Touchdown: step %d/%d, descended %.1fmm, "
                              "|dFz|=%.2fN", i, n_steps, descended * 1000,
                              abs(self._get_wrench_z() - baseline))

        rospy.sleep(0.2)  # final settle
        final_pose = self._group.get_current_pose().pose
        actual_z = final_pose.position.z
        # Raw delta: how much the TCP moved relative to APPROACH start.
        # This will always be ≤ 0 (probe descended).
        dz_raw = actual_z - planned_z
        # Convert to scan-applicable correction: add back approach_height,
        # because the planned scan waypoints are at WP_Z (= surface
        # estimate), not at WP_Z + approach_height.
        dz_for_scan = dz_raw + self.approach_height

        if not contact_detected:
            rospy.logwarn(
                "Touchdown: no contact within %.0fmm — surface may be "
                "lower than expected, F/T not zeroed, or force "
                "threshold too high.", self.touchdown_max_descent * 1000)
            return None

        rospy.loginfo(
            "Touchdown: contact at tool0_z=%.4f (started at %.4f). "
            "Raw descent=%+.1fmm; scan correction (+approach_height) "
            "=%+.1fmm  →  real surface is %s the depth-cam estimate "
            "by %.1fmm.",
            actual_z, planned_z, dz_raw * 1000, dz_for_scan * 1000,
            "BELOW" if dz_for_scan < 0 else "ABOVE",
            abs(dz_for_scan * 1000))

        # Optionally press a bit further (compensates for any settling /
        # ensures consistent contact force across the line).
        if self.touchdown_extra_press > 0:
            extra_target = Pose()
            extra_target.position.x = final_pose.position.x
            extra_target.position.y = final_pose.position.y
            extra_target.position.z = (final_pose.position.z -
                                       self.touchdown_extra_press)
            extra_target.orientation = final_pose.orientation
            self._reset_start_state()
            try:
                p, f = self._group.compute_cartesian_path(
                    [extra_target], 0.001, 0.0, avoid_collisions=False)
            except TypeError:
                p, f = self._group.compute_cartesian_path(
                    [extra_target], 0.001, 0.0, False)
            if f > 0.9:
                self._group.execute(p, wait=True)
                rospy.sleep(0.2)
                final_pose = self._group.get_current_pose().pose
                dz_raw = final_pose.position.z - planned_z
                dz_for_scan = dz_raw + self.approach_height
                rospy.loginfo("Touchdown: pressed extra %.1fmm, "
                              "scan correction now %+.1fmm",
                              self.touchdown_extra_press * 1000,
                              dz_for_scan * 1000)
        return dz_for_scan

    # ── MoveIt start-state reset ─────────────────────────────────

    def _reset_start_state(self):
        """Reset MoveGroup planning start state to the current robot
        state. The convenience method ``set_start_state_to_current_value``
        is missing from older python-moveit-commander wheels (Noetic
        ships several, behaviour varies). Try it first; fall back to
        explicit ``set_start_state(get_current_state())`` from
        RobotCommander, then to a no-op (planning will then default to
        the current state on the next call anyway)."""
        try:
            if hasattr(self._group, 'set_start_state_to_current_value'):
                self._group.set_start_state_to_current_value()
                return
        except Exception as e:
            rospy.logdebug("set_start_state_to_current_value failed: %s", e)
        try:
            if not hasattr(self, '_robot'):
                self._robot = moveit_commander.RobotCommander()
            self._group.set_start_state(self._robot.get_current_state())
            return
        except Exception as e:
            rospy.logdebug("set_start_state(get_current_state) failed: %s", e)
        # Last resort: clear start state so the next plan() falls back
        # to the live robot state automatically.
        try:
            self._group.set_start_state(moveit_commander.RobotCommander()
                                        .get_current_state())
        except Exception:
            pass

    def _plan_cartesian_scan(self, cartesian_poses, avoid_collisions=False):
        """Plan the Cartesian scan path from current state.

        Limits BOTH velocity AND acceleration. Without an explicit
        acceleration cap MoveIt time-parameterizes the trajectory at
        the planning-group default (often 1.0 = no limit), which
        produces sharp accel spikes at trajectory endpoints. Those
        spikes trip the UR's joint-acceleration safety (error
        C153Ax → PROTECTIVE_STOP) when the probe is in contact with
        a surface that resists the requested motion.
        Call AFTER approach.

        avoid_collisions defaults to False because the probe is meant
        to be in contact with the body surface — and the body shows up
        in the planning scene as Octomap voxels (built from the depth
        camera point cloud). With collision checking on, MoveIt will
        refuse to plan through those voxels and the Cartesian path
        truncates at ~10% (the moment the probe enters the body).

        IMPORTANT: always reset the planning start state to the current
        robot state before computing the Cartesian path. Otherwise
        compute_cartesian_path uses a stale start state (e.g. left
        over from a previous preview/plan call), and the resulting
        plan's first joint angles differ from the live robot state,
        causing the trajectory controller to reject the goal with
        "Invalid Trajectory: start point deviates from current robot
        state".
        """
        self._reset_start_state()
        speed_factor = min(1.0, self.scan_speed / 0.25)
        self._group.set_max_velocity_scaling_factor(speed_factor)
        # Cap acceleration aggressively — the contact phase of scan
        # is extremely sensitive to trajectory acceleration spikes.
        accel_factor = rospy.get_param('/scan/scan_accel_factor', 0.05)
        self._group.set_max_acceleration_scaling_factor(accel_factor)
        try:
            scan_plan, fraction = self._group.compute_cartesian_path(
                cartesian_poses, 0.005, 0.0,
                avoid_collisions=avoid_collisions)
        except TypeError:
            # Older MoveIt python bindings don't accept avoid_collisions
            # as a keyword — fall back to positional. The 4th positional
            # arg is avoid_collisions in MoveIt 1.x.
            scan_plan, fraction = self._group.compute_cartesian_path(
                cartesian_poses, 0.005, 0.0, avoid_collisions)

        # ── Retime trajectory with explicit Cartesian speed ───────
        # compute_cartesian_path returns a geometric path WITHOUT
        # proper time parameterization — set_max_velocity_scaling_factor
        # is NOT applied automatically. We need to re-time the
        # trajectory so the actual TCP linear velocity matches
        # scan_speed (otherwise the scan runs at default joint
        # velocity, ~30-100 mm/s, which is way too fast for capture).
        if scan_plan and len(scan_plan.joint_trajectory.points) >= 2:
            scan_plan = self._retime_trajectory(scan_plan, speed_factor,
                                                 accel_factor)
        return scan_plan, fraction

    def _retime_trajectory(self, plan, velocity_scaling, accel_scaling):
        """Re-time a planned trajectory so the actual TCP velocity
        matches scan_speed. Required because compute_cartesian_path
        does NOT apply set_max_velocity_scaling_factor — it returns
        a geometric path with default (often very fast) timing.

        Tries several approaches in order:
          1. MoveGroupCommander.retime_trajectory(ref_state, plan,
             velocity_scaling_factor=..., acceleration_scaling_factor=...)
             [Newer MoveIt versions]
          2. Same call without keyword args (positional only)
             [Older versions where kwargs aren't accepted]
          3. Manually scale time_from_start of every trajectory point
             by 1/velocity_scaling
             [Always works as a last resort]
        """
        n_pts = len(plan.joint_trajectory.points)
        old_dur = (plan.joint_trajectory.points[-1].time_from_start.to_sec()
                   if n_pts >= 2 else 0)
        rospy.loginfo("Retime input: %d pts, duration %.3fs, "
                      "velocity_scaling=%.4f, accel_scaling=%.4f",
                      n_pts, old_dur, velocity_scaling, accel_scaling)

        # ── Attempt 1: MoveGroup retime_trajectory with keywords
        try:
            if not hasattr(self, '_robot'):
                self._robot = moveit_commander.RobotCommander()
            ref_state = self._robot.get_current_state()
            retimed = self._group.retime_trajectory(
                ref_state, plan,
                velocity_scaling_factor=velocity_scaling,
                acceleration_scaling_factor=accel_scaling)
            new_dur = (retimed.joint_trajectory.points[-1].time_from_start.to_sec()
                       if len(retimed.joint_trajectory.points) >= 2 else 0)
            if new_dur > old_dur * 1.5 or new_dur > 5.0:
                rospy.loginfo(
                    "Retime SUCCESS (kwargs): %.3fs → %.3fs", old_dur, new_dur)
                return retimed
            else:
                rospy.logwarn(
                    "Retime kwargs returned suspicious dur=%.3fs "
                    "(unchanged?); trying manual scale", new_dur)
        except TypeError as e:
            rospy.logwarn("Retime kwargs TypeError: %s", e)
        except Exception as e:
            rospy.logwarn("Retime kwargs raised: %s: %s",
                          type(e).__name__, e)

        # ── Attempt 2: Positional args
        try:
            retimed = self._group.retime_trajectory(
                ref_state, plan, velocity_scaling, accel_scaling)
            new_dur = (retimed.joint_trajectory.points[-1].time_from_start.to_sec()
                       if len(retimed.joint_trajectory.points) >= 2 else 0)
            if new_dur > old_dur * 1.5 or new_dur > 5.0:
                rospy.loginfo(
                    "Retime SUCCESS (positional): %.3fs → %.3fs",
                    old_dur, new_dur)
                return retimed
        except Exception as e:
            rospy.logwarn("Retime positional failed: %s", e)

        # ── Attempt 3: Manual scaling (always works)
        # Just multiply every time_from_start by (1/velocity_scaling).
        # Doesn't optimise jerk/accel limits but guarantees the
        # trajectory takes the requested time.
        try:
            from copy import deepcopy
            retimed = deepcopy(plan)
            scale = 1.0 / max(velocity_scaling, 1e-6)
            for pt in retimed.joint_trajectory.points:
                pt.time_from_start = rospy.Duration(
                    pt.time_from_start.to_sec() * scale)
                # Also scale velocities/accelerations down proportionally
                if pt.velocities:
                    pt.velocities = [v / scale for v in pt.velocities]
                if pt.accelerations:
                    pt.accelerations = [a / (scale * scale)
                                        for a in pt.accelerations]
            new_dur = (retimed.joint_trajectory.points[-1].time_from_start.to_sec()
                       if len(retimed.joint_trajectory.points) >= 2 else 0)
            rospy.loginfo(
                "Retime SUCCESS (manual scale): %.3fs → %.3fs (scale=%.1fx)",
                old_dur, new_dur, scale)
            return retimed
        except Exception as e:
            rospy.logerr("Retime manual scale failed (%s) — scan WILL "
                         "run at default fast speed!", e)
            return plan

    def _plan_cartesian_with_fallback(self, keypoints, scan_orientation,
                                      dz_correction=0.0,
                                      override_z=None,
                                      min_probe_tip_z=None,
                                      min_fraction=0.5):
        """Try per-waypoint orientation first; if Cartesian path is too
        infeasible, retry with a single shared orientation.

        Returns (plan, fraction, used_shared_orientation).
        """
        # Attempt 1: per-waypoint orientations
        cartesian_poses = self._build_cartesian_poses(
            keypoints, orientation=scan_orientation,
            use_planner_orientation=True,
            dz_correction=dz_correction,
            override_z=override_z,
            min_probe_tip_z=min_probe_tip_z)
        plan, fraction = self._plan_cartesian_scan(cartesian_poses)
        if fraction >= min_fraction:
            return plan, fraction, False

        rospy.logwarn(
            "Per-waypoint orientations gave only %.0f%% Cartesian "
            "feasibility — retrying with shared orientation.",
            fraction * 100)

        # Attempt 2: shared orientation
        cartesian_poses = self._build_cartesian_poses(
            keypoints, orientation=scan_orientation,
            use_planner_orientation=False,
            dz_correction=dz_correction,
            override_z=override_z,
            min_probe_tip_z=min_probe_tip_z)
        plan, fraction = self._plan_cartesian_scan(cartesian_poses)
        return plan, fraction, True

    def _preview_scan_srv(self, req):
        """Plan the scan path and display in RViz without executing."""
        if self.scan_path is None or len(self.scan_path.poses) == 0:
            return TriggerResponse(success=False, message="No scan path available")

        try:
            lines = self._split_into_lines(self.scan_path.poses)
            rospy.loginfo("Preview: %d scan lines", len(lines))

            all_ok = True
            for line_idx, line_wps in enumerate(lines):
                keypoints = self._subsample_waypoints(line_wps)
                # Pick the smaller-wrist-rotation equivalent of the
                # planner's orientations (180°-around-probe-Z flip).
                keypoints = self._maybe_flip_path_orientations(keypoints)
                # Use the planner's scan orientation (probe long-axis
                # perpendicular to scan path) rather than the current
                # pose orientation.
                scan_orientation = self._pick_scan_orientation(keypoints)
                approach_pose = self._build_approach_pose(
                    keypoints, scan_orientation)

                # Plan approach to get the joint state at approach position
                self._group.set_max_velocity_scaling_factor(0.1)
                self._group.set_pose_target(approach_pose)
                approach_result = self._group.plan()
                self._group.clear_pose_targets()

                if isinstance(approach_result, tuple):
                    approach_ok, approach_traj = approach_result[0], approach_result[1]
                else:
                    approach_ok = True
                    approach_traj = approach_result

                if not approach_ok or len(approach_traj.joint_trajectory.points) == 0:
                    rospy.logwarn("Line %d: approach planning failed", line_idx + 1)
                    all_ok = False
                    continue

                # Set MoveGroup start state to approach end-state, then plan Cartesian
                approach_end_joints = approach_traj.joint_trajectory.points[-1].positions
                start_state = self._group.get_current_state()
                joint_names = approach_traj.joint_trajectory.joint_names
                for i, name in enumerate(joint_names):
                    idx = start_state.joint_state.name.index(name)
                    pos_list = list(start_state.joint_state.position)
                    pos_list[idx] = approach_end_joints[i]
                    start_state.joint_state.position = pos_list
                self._group.set_start_state(start_state)

                # Plan Cartesian path with automatic fallback from
                # per-waypoint orientations to shared orientation when
                # the surface-following orientations would be too
                # aggressive for IK. Preview uses dz_correction=0
                # (no touchdown happens during preview).
                scan_plan, fraction, used_shared = \
                    self._plan_cartesian_with_fallback(
                        keypoints, scan_orientation,
                        dz_correction=0.0)

                # Reset start state to current (API differs across MoveIt
                # versions: prefer set_start_state_to_current_value when
                # available, fall back to passing current state explicitly).
                self._reset_start_state()

                if fraction < 0.5:
                    rospy.logwarn("Line %d: Cartesian path only %.0f%% feasible "
                                  "(%d keypoints, %s orientation)",
                                  line_idx + 1, fraction * 100,
                                  len(keypoints),
                                  "shared" if used_shared else "per-waypoint")
                    all_ok = False
                    continue

                rospy.loginfo("Line %d: %d keypoints, %.0f%% Cartesian coverage%s",
                              line_idx + 1, len(keypoints), fraction * 100,
                              " (shared orientation)" if used_shared else
                              " (per-waypoint orientations)")

                display = DisplayTrajectory()
                display.trajectory_start = start_state
                display.trajectory.append(approach_traj)
                display.trajectory.append(scan_plan)
                self.pub_display_traj.publish(display)

            self._preview_plan = lines
            msg = "Preview ready (%d lines). Check RViz, then call /us3d/start_scan to execute." % len(lines)
            return TriggerResponse(success=all_ok, message=msg)

        except Exception as e:
            rospy.logerr("Preview failed: %s", str(e))
            self._reset_start_state()
            return TriggerResponse(success=False, message="Preview failed: %s" % str(e))

    def _check_controller_running(self, controller='scaled_pos_joint_traj_controller'):
        """Verify the trajectory controller is in 'running' state.

        When the UR ExternalControl program on the teach pendant is
        stopped, the controller_stopper node kills the trajectory
        controller and any motion request fails with INVALID_GOAL after
        ~10s of planning. Catching this up front saves the user time
        and gives a clear error message.
        """
        try:
            from controller_manager_msgs.srv import ListControllers
            srv = rospy.ServiceProxy(
                '/controller_manager/list_controllers', ListControllers)
            srv.wait_for_service(timeout=2.0)
            resp = srv()
            for ctrl in resp.controller:
                if ctrl.name == controller:
                    return ctrl.state == 'running', ctrl.state
            return False, 'not_loaded'
        except Exception as e:
            rospy.logwarn("Controller check failed: %s", e)
            return True, 'unknown'  # don't block on a check failure

    def _start_scan_srv(self, req):
        if self.scan_path is None or len(self.scan_path.poses) == 0:
            return TriggerResponse(success=False, message="No scan path available")
        if self.scanning:
            return TriggerResponse(success=False, message="Already scanning")

        ok, state = self._check_controller_running()
        if not ok:
            msg = (
                "Controller scaled_pos_joint_traj_controller is '%s' (need 'running'). "
                "Press PLAY on the teach pendant to start the ExternalControl "
                "program, then retry /us3d/start_scan." % state)
            rospy.logerr(msg)
            return TriggerResponse(success=False, message=msg)

        self.scanning = True
        rospy.loginfo("Starting scan (MoveIt Cartesian path)...")

        try:
            lines = self._split_into_lines(self.scan_path.poses)
            for line_idx, line_wps in enumerate(lines):
                if not self.scanning:
                    rospy.logwarn("Scan aborted by user")
                    break
                rospy.loginfo("Scanning line %d/%d (%d waypoints)",
                              line_idx + 1, len(lines), len(line_wps))
                self._execute_scan_line(line_wps)

            return TriggerResponse(success=True, message="Scan completed successfully")
        except Exception as e:
            rospy.logerr("Scan failed: %s", str(e))
            self._group.stop()
            return TriggerResponse(success=False, message="Scan failed: %s" % str(e))
        finally:
            self.scanning = False

    def _stop_scan_srv(self, req):
        self.scanning = False
        self._group.stop()
        return TriggerResponse(success=True, message="Scan stopped")

    def _execute_scan_force_adaptive(self, keypoints, scan_orientation,
                                      dz_correction, override_z,
                                      contact_probe_z):
        """Execute scan path SEGMENT-BY-SEGMENT with mid-scan Z adjustment.

        After each segment, read current Fz. If |Fz| < min_force,
        contact is lost — lower the Z target by `step` mm for all
        subsequent segments. This is a poor-man's force control: no
        true forceMode, but allows the path to adapt downward when the
        depth-cam-predicted surface was too high.

        z_offset is monotonically non-increasing (only pushes deeper,
        never lifts back up). Capped at -max_push to prevent runaway.
        """
        baseline = (self._wrench_z_baseline
                    if hasattr(self, '_wrench_z_baseline') else 0.0)
        z_offset = 0.0   # additional downward shift, m (≤ 0)

        rospy.loginfo(
            "Force-adaptive scan: %d segments, "
            "min |Fz|=%.1fN (else +%.1fmm down), max_push=%.1fmm",
            len(keypoints) - 1,
            self.force_adaptive_min_force,
            self.force_adaptive_step * 1000,
            self.force_adaptive_max_push * 1000)

        for i in range(1, len(keypoints)):
            target_kp = keypoints[i]

            # Build single-target Cartesian segment from CURRENT pose
            # to target_kp. Apply running z_offset to ALL Z controls
            # (target Z, dz_correction, override_z, safety floor).
            if override_z is not None:
                seg_override_z = override_z + z_offset
            else:
                seg_override_z = None
            seg_dz = dz_correction + z_offset
            # Safety floor moves DOWN with z_offset (must allow what
            # we're about to command).
            seg_min_z = (contact_probe_z + z_offset
                         if contact_probe_z is not None else None)

            seg_poses = self._build_cartesian_poses(
                [target_kp], orientation=scan_orientation,
                use_planner_orientation=True,
                dz_correction=seg_dz,
                override_z=seg_override_z,
                min_probe_tip_z=seg_min_z)

            seg_plan, frac = self._plan_cartesian_scan(seg_poses)
            if frac < 0.5:
                rospy.logwarn(
                    "Adaptive segment %d/%d: only %.0f%% feasible, skipping",
                    i, len(keypoints) - 1, frac * 100)
                continue

            self._group.execute(seg_plan, wait=True)
            self._group.stop()

            # Settle then read force
            rospy.sleep(self.force_adaptive_settle_s)
            fz_now = self._get_wrench_z()
            d_fz = abs(fz_now - baseline)

            if d_fz < self.force_adaptive_min_force:
                # Lost contact — push deeper for next segment(s)
                new_offset = z_offset - self.force_adaptive_step
                new_offset = max(new_offset, -self.force_adaptive_max_push)
                if new_offset < z_offset:
                    rospy.loginfo(
                        "Segment %d/%d: |dFz|=%.2fN < %.1fN, "
                        "lowering Z by %.1fmm (cumulative offset %.1fmm)",
                        i, len(keypoints) - 1, d_fz,
                        self.force_adaptive_min_force,
                        self.force_adaptive_step * 1000,
                        new_offset * 1000)
                    z_offset = new_offset
                else:
                    rospy.logwarn(
                        "Segment %d/%d: |dFz|=%.2fN but Z already at "
                        "max push -%.1fmm, no further adjustment.",
                        i, len(keypoints) - 1, d_fz,
                        self.force_adaptive_max_push * 1000)

        rospy.loginfo(
            "Force-adaptive scan complete. Final z_offset=%+.1fmm",
            z_offset * 1000)

    def _execute_scan_line(self, waypoints):
        """Execute one scan line.

        Sequence:
          1. Move above scan start (approach)
          2. Hide cursor + zero F/T (probe still in air)
          3. Descend to first scan waypoint (probe touches surface)
          4. STABLE WAIT before recording → contact settles, no transient frames
          5. Auto-start recording
          6. Plan + execute Cartesian scan
          7. STABLE WAIT after recording → no retract motion in dataset
          8. Auto-stop recording
          9. Retract above scan end
        """
        keypoints = self._subsample_waypoints(waypoints)
        probe_z_offset = self._get_probe_z_offset()
        current_pose = self._group.get_current_pose().pose

        # Diagnostic: where does the planner ACTUALLY want the probe_tip
        # to go? Useful when the robot appears to overshoot or deviate.
        if len(keypoints) >= 2:
            kp0 = keypoints[0].position
            kpN = keypoints[-1].position
            scan_len_mm = float(np.linalg.norm(np.array([
                kpN.x - kp0.x, kpN.y - kp0.y, kpN.z - kp0.z]))) * 1000
            rospy.loginfo(
                "Planner waypoint endpoints (probe_tip targets):\n"
                "  start: (%.4f, %.4f, %.4f)\n"
                "  end:   (%.4f, %.4f, %.4f)\n"
                "  total length=%.1fmm  (#keypoints=%d, "
                "#waypoints=%d)",
                kp0.x, kp0.y, kp0.z, kpN.x, kpN.y, kpN.z,
                scan_len_mm, len(keypoints), len(waypoints))

        # ── Resolve the orientation ambiguity:
        #    The planner picks probe_x = scan_dir, but probe_x = -scan_dir
        #    is equally valid (just rotates the probe 180° around its
        #    own Z axis — physically identical scan, mirrored image).
        #    Pick whichever requires LESS wrist rotation from current.
        keypoints = self._maybe_flip_path_orientations(keypoints)

        # ── Pick the scan orientation from the planner's first waypoint
        #    (probe perpendicular to surface, long-axis perpendicular to
        #    the scan direction). The wrist rotates to this attitude
        #    while still suspended above the surface — it's much safer
        #    to do the rotation in mid-air than while in contact.
        scan_orientation = self._pick_scan_orientation(keypoints)
        wrist_angle = self._angle_between_quats(
            current_pose.orientation, scan_orientation)
        wrist_deg = np.degrees(wrist_angle)
        rospy.loginfo(
            "Scan attitude: rotating wrist by %.1f° from current pose "
            "(probe long-axis ⟂ scan path)", wrist_deg)
        if wrist_deg > 120:
            rospy.logwarn(
                "Wrist rotation %.0f° is large — Cartesian SLERP may "
                "produce high joint speeds at the start of motion, "
                "raising the risk of C153A* (joint accel) protective "
                "stop. Consider using /us3d/record_home to set a home "
                "pose closer to the scan direction (e.g. wrist already "
                "aligned with marker0→marker1 axis).", wrist_deg)

        # 1. Move to approach position above scan start, ALREADY at the
        #    target scan orientation. This single MoveIt goal both
        #    translates the TCP to above the scan start AND rotates the
        #    wrist to the proper scan attitude. Uses _move_to_pose
        #    which clears octomap first and falls back to Cartesian if
        #    a fresh octomap voxel pops up mid-execution.
        approach_pose = self._build_approach_pose(
            keypoints, scan_orientation, probe_z_offset)
        rospy.loginfo("Approach to (%.3f, %.3f, %.3f) at scan orientation",
                      approach_pose.position.x, approach_pose.position.y,
                      approach_pose.position.z)

        if not self._move_to_pose(approach_pose, vel_scale=0.1):
            rospy.logwarn("Failed to reach approach position, skipping line")
            return

        # 2. Probe is suspended above surface — safe place to hide cursor
        #    and zero the F/T sensor before contact.
        self._hide_cursor()
        self._zero_ft_sensor()
        rospy.sleep(0.5)   # let zero settle before reading wrench

        # 2b. Force-based Z calibration: descend until contact, record
        #     the offset between the depth-camera-predicted surface
        #     height and the actual contact height. This is critical
        #     when the depth cloud disagrees with the real surface
        #     (typical RGBD error 1-5cm on glossy/uniform tissue).
        dz_correction = self._touchdown_calibrate(scan_orientation) or 0.0
        if dz_correction != 0.0:
            rospy.loginfo("Applying Z correction of %+.1fmm to all "
                          "waypoints", dz_correction * 1000)

        # 3. Build Cartesian scan poses.
        #
        # Decide between two scan-Z strategies:
        #
        #   (A) FLAT scan at touchdown contact Z (recommended,
        #       default true): every waypoint has the same probe_tip Z
        #       = whatever Z the touchdown actually reached. This
        #       completely sidesteps the depth-camera Z curve, which
        #       on sparse point clouds (often <100 points after crop)
        #       can have 20-30mm of RBF artifacts that drive the
        #       probe up/down during scan and trigger PROTECTIVE_STOP
        #       when the probe collides with the surface.
        #
        #   (B) Per-waypoint Z + dz_correction: trust the planner's
        #       Z curve, just shift it by the touchdown offset. Use
        #       this only when scanning a genuinely curved surface
        #       AND your depth cloud is dense enough to produce a
        #       smooth surface fit.
        flat_scan = rospy.get_param('/scan/flat_scan_at_contact_z', False)
        override_z = None
        min_probe_tip_z = None

        # ── Optional: invert the per-waypoint Z trend ───────────────
        # Use this when you observe in RViz that the depth-camera-
        # derived path Z trend goes the OPPOSITE direction of the
        # actual surface (e.g. path Z descends but real surface
        # rises). Reflects every waypoint Z about keypoints[0].z so
        # the slope sign flips. Same XY positions, opposite Z slope.
        invert_z = rospy.get_param('/scan/invert_z_trend', False)
        if invert_z and len(keypoints) > 1:
            z0 = keypoints[0].position.z
            for wp in keypoints:
                wp.position.z = 2.0 * z0 - wp.position.z   # reflect about z0
            rospy.loginfo(
                "INVERTED Z trend: per-waypoint Z deviations reflected "
                "about keypoints[0].z=%.4f m. Use when depth-cam Z "
                "slope points the wrong way relative to real surface.",
                z0)

        # Read the actual contact Z right after touchdown so we can use
        # it as either an override (flat scan) OR a safety floor.
        contact_probe_z = None
        try:
            tf_p = self._tf_buffer.lookup_transform(
                'base_link', 'probe_tip', rospy.Time(0),
                rospy.Duration(0.5))
            contact_probe_z = tf_p.transform.translation.z
        except Exception as e:
            rospy.logwarn("Couldn't read probe_tip TF (%s)", e)

        if contact_probe_z is not None and dz_correction != 0.0:
            if flat_scan:
                # Force EVERY waypoint to the contact Z
                override_z = contact_probe_z
                rospy.loginfo(
                    "Flat-scan: forcing probe_tip.z=%.4f m (contact "
                    "height) for ALL waypoints, ignoring depth-cam "
                    "Z curve.", override_z)
            else:
                # Safety floor: probe can lift up freely (depth-cam
                # may say surface rises) but it CANNOT be commanded
                # to push deeper than the touchdown contact height
                # plus a small allowed extra-press.
                max_push_mm = rospy.get_param(
                    '/scan/max_push_below_contact_mm', 0.0)
                min_probe_tip_z = contact_probe_z - max_push_mm / 1000.0
                rospy.loginfo(
                    "Z safety floor: probe_tip will NOT be commanded "
                    "below %.4f m (contact_z %.4f m − max_push %.1fmm). "
                    "Probe can lift freely above this; depth-cam Z "
                    "curve drives the path above the floor.",
                    min_probe_tip_z, contact_probe_z, max_push_mm)

        scan_plan, fraction, used_shared = self._plan_cartesian_with_fallback(
            keypoints, scan_orientation,
            dz_correction=dz_correction,
            override_z=override_z,
            min_probe_tip_z=min_probe_tip_z)

        rospy.loginfo("Cartesian scan: %d keypoints, %.0f%% feasible%s",
                      len(keypoints), fraction * 100,
                      " (shared orientation)" if used_shared else
                      " (per-waypoint orientations)")

        # Diagnostic: dump the actual joint trajectory endpoints, to
        # verify the trajectory ends where we think it should.
        if scan_plan and len(scan_plan.joint_trajectory.points) >= 2:
            try:
                # Compute forward kinematics for first and last point
                from sensor_msgs.msg import JointState
                jt = scan_plan.joint_trajectory
                # Just print joint values to compare manually if needed
                first_jt = jt.points[0].positions
                last_jt = jt.points[-1].positions
                rospy.loginfo(
                    "Trajectory joint endpoints (should map to first/last "
                    "scan keypoint via FK):\n"
                    "  joints[0] = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n"
                    "  joints[N] = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                    *first_jt, *last_jt)
            except Exception:
                pass

        # ── Pre-scan diagnostic ─────────────────────────────────
        # Log: planned probe_tip Z (= waypoint Z + dz_correction) vs
        #      actual probe_tip Z right now (from TF). At this point
        #      the robot is at the approach pose (probe is 5cm = the
        #      `approach_height` above the planned surface — that is
        #      BY DESIGN). After Cartesian execution starts, the
        #      probe_tip should descend to wp.z + dz_correction.
        try:
            tf_p = self._tf_buffer.lookup_transform(
                'base_link', 'probe_tip', rospy.Time(0), rospy.Duration(0.5))
            actual_probe_z = tf_p.transform.translation.z
            target_probe_z_first = (keypoints[0].position.z + dz_correction)
            target_probe_z_last = (keypoints[-1].position.z + dz_correction)
            rospy.loginfo(
                "Scan geometry check:\n"
                "  approach_height (in air):  %+.1f mm\n"
                "  actual probe_tip.z now:    %.4f m\n"
                "  target probe_tip.z (first): %.4f m   (Δ=%+.1f mm — should ≈ -approach_height)\n"
                "  target probe_tip.z (last):  %.4f m\n"
                "  dz_correction applied:     %+.1f mm",
                self.approach_height * 1000, actual_probe_z,
                target_probe_z_first,
                (target_probe_z_first - actual_probe_z) * 1000,
                target_probe_z_last,
                dz_correction * 1000)
        except Exception:
            pass

        if fraction < 0.5:
            rospy.logwarn("Cartesian path only %.0f%% feasible, skipping line",
                          fraction * 100)
            return

        # 4. Settle before recording starts.
        #    The first servo cycles after planner switch produce micro-jitter
        #    that we don't want polluting the dataset.
        if self.stable_wait_before > 0:
            rospy.loginfo("Settling for %.2fs before recording...",
                          self.stable_wait_before)
            rospy.sleep(self.stable_wait_before)

        # 5. Start recording AFTER settle so no transient frames are captured.
        self._start_recording()

        # 6. Execute scan
        # ★ UR speed slider is set ONLY around the actual execute()
        #   call, not for approach/touchdown/retract. This way only
        #   the contact-scan portion runs slow; non-contact moves
        #   stay at full speed. try/finally guarantees restore even
        #   if execute fails, raises, or controller drops out.
        speed_slider_used = self._set_speed_slider_for_scan()
        try:
            if self.force_adaptive_scan:
                # Per-segment execution with mid-scan Z adjustment
                self._execute_scan_force_adaptive(
                    keypoints, scan_orientation, dz_correction,
                    override_z, contact_probe_z)
            else:
                # Single shot: pre-planned scan path
                self._group.execute(scan_plan, wait=True)
                self._group.stop()
        finally:
            if speed_slider_used is not None:
                self._restore_speed_slider()

        # 7. Hold still before stopping recording — guarantees the dataset
        #    ends with the probe stationary at the last waypoint, not mid-retract.
        if self.stable_wait_after > 0:
            rospy.loginfo("Holding for %.2fs before stopping recording...",
                          self.stable_wait_after)
            rospy.sleep(self.stable_wait_after)

        # 8. Stop recording BEFORE retract so the lift-off frames (in air,
        #    bright air-interface reflection, falling Fz) are excluded.
        self._stop_recording()

        # ── Post-scan diagnostic ────────────────────────────────
        try:
            tf_p = self._tf_buffer.lookup_transform(
                'base_link', 'probe_tip', rospy.Time(0), rospy.Duration(0.5))
            actual_probe_z_end = tf_p.transform.translation.z
            target_probe_z_end = (keypoints[-1].position.z + dz_correction)
            err_mm = (actual_probe_z_end - target_probe_z_end) * 1000
            rospy.loginfo(
                "Post-scan: probe_tip.z actual=%.4f, target=%.4f, Δ=%+.1f mm",
                actual_probe_z_end, target_probe_z_end, err_mm)
            if abs(err_mm) > 5.0:
                rospy.logwarn(
                    "Probe Z error %+.1f mm exceeds 5 mm. "
                    "Possible causes: (1) Cartesian path didn't converge "
                    "to last waypoint, (2) controller couldn't track, "
                    "(3) probe collided with surface and stalled.",
                    err_mm)
        except Exception:
            pass

        # 9. Retract above scan end (apply same Z correction so we
        #    actually lift OFF the surface, not just to the planned
        #    surface height which may be 5cm off). Use _move_to_pose
        #    so octomap clears + Cartesian fallback also apply here.
        retract_pose = Pose()
        retract_pose.position.x = keypoints[-1].position.x
        retract_pose.position.y = keypoints[-1].position.y
        retract_pose.position.z = (keypoints[-1].position.z + probe_z_offset
                                   + dz_correction + self.approach_height)
        retract_pose.orientation = scan_orientation
        self._move_to_pose(retract_pose, vel_scale=0.1)

    def _split_into_lines(self, poses):
        if len(poses) <= 1:
            return [poses]
        lines = []
        current_line = [poses[0]]
        threshold = rospy.get_param('/scan/line_spacing', 0.003) * 0.5
        for i in range(1, len(poses)):
            p1 = poses[i - 1].position
            p2 = poses[i].position
            dist = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)
            if dist > threshold:
                lines.append(current_line)
                current_line = [poses[i]]
            else:
                current_line.append(poses[i])
        if current_line:
            lines.append(current_line)
        return lines

    def _subsample_waypoints(self, waypoints):
        """Pick keypoints for Cartesian path (every ~15mm)."""
        if len(waypoints) <= 2:
            return waypoints
        step_m = rospy.get_param('/scan/keypoint_step', 0.015)
        wp_step = rospy.get_param('/scan/waypoint_step', 0.001)
        stride = max(1, int(step_m / wp_step))
        keypoints = [waypoints[0]]
        for i in range(stride, len(waypoints) - 1, stride):
            keypoints.append(waypoints[i])
        if keypoints[-1] is not waypoints[-1]:
            keypoints.append(waypoints[-1])
        return keypoints

    # ── Path orientation pre-processing ──────────────────────────

    def _flip_quat_180_around_local_z(self, q_msg):
        """Return q_msg ⊗ Rz(180°). Equivalent to rotating the local
        frame 180° around its own Z axis: probe_z stays the same
        (tip still points toward surface), probe_x → -probe_x,
        probe_y → -probe_y.

        For freehand 3D US this is a physically equivalent scan: the
        US image is mirrored 180° but the slab geometry (slice
        positions and normals) is identical. Useful for picking
        whichever of the two ambiguous waypoint orientations
        requires LESS wrist rotation from the current pose.
        """
        # q_msg as ROS Quaternion (w,x,y,z order), convert to wxyz np
        q = np.array([q_msg.w, q_msg.x, q_msg.y, q_msg.z])
        # Rz(180°) = [w=0, x=0, y=0, z=1]
        rz180 = np.array([0.0, 0.0, 0.0, 1.0])
        q_new = tq.qmult(q, rz180)
        out = type(q_msg)()
        out.w = float(q_new[0])
        out.x = float(q_new[1])
        out.y = float(q_new[2])
        out.z = float(q_new[3])
        return out

    def _maybe_flip_path_orientations(self, waypoints):
        """If the planner's chosen waypoint orientations would require
        the wrist to rotate >90° from the current pose, flip ALL
        waypoint orientations 180° around the probe-Z axis. Both
        choices represent the same physical scan (probe still points
        the same way at the surface; only the in-plane lateral
        direction reverses), but one choice may need a 174° wrist
        swing while the other needs only 6°.
        """
        if not waypoints:
            return waypoints

        current_q = self._group.get_current_pose().pose.orientation
        first_q = waypoints[0].orientation
        n2 = first_q.x ** 2 + first_q.y ** 2 + first_q.z ** 2 + first_q.w ** 2
        if abs(n2 - 1.0) > 0.05:
            return waypoints  # invalid quat, leave alone

        ang_orig = np.degrees(self._angle_between_quats(current_q, first_q))
        flipped_first = self._flip_quat_180_around_local_z(first_q)
        ang_flip = np.degrees(self._angle_between_quats(
            current_q, flipped_first))

        if ang_flip + 1.0 >= ang_orig:
            # Original is already (close to) better; no flip
            rospy.loginfo(
                "Path orientation: keeping planner's choice "
                "(wrist rot %.1f° vs %.1f° if flipped)",
                ang_orig, ang_flip)
            return waypoints

        rospy.loginfo(
            "Path orientation: flipping all waypoints 180° around "
            "probe-Z (wrist rot %.1f° → %.1f°). Physically equivalent "
            "scan; image is mirrored but slab geometry is unchanged.",
            ang_orig, ang_flip)

        # Flip every waypoint orientation by the same 180°-Z so the
        # whole sequence stays smooth.
        flipped_wps = []
        for wp in waypoints:
            new_wp = Pose()
            new_wp.position.x = wp.position.x
            new_wp.position.y = wp.position.y
            new_wp.position.z = wp.position.z
            new_wp.orientation = self._flip_quat_180_around_local_z(
                wp.orientation)
            flipped_wps.append(new_wp)
        return flipped_wps

    @staticmethod
    def _to_geometry_pose(ros_pose):
        """Convert a geometry_msgs/Pose to a fresh copy."""
        p = Pose()
        p.position.x = ros_pose.position.x
        p.position.y = ros_pose.position.y
        p.position.z = ros_pose.position.z
        p.orientation.x = ros_pose.orientation.x
        p.orientation.y = ros_pose.orientation.y
        p.orientation.z = ros_pose.orientation.z
        p.orientation.w = ros_pose.orientation.w
        return p


if __name__ == '__main__':
    try:
        node = ForceScanNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
