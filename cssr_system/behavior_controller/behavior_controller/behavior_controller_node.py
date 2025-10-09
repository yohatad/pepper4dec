#!/usr/bin/env python3
"""
Tour Guide BehaviorTree ROS2 Node (Humble)
Uses py_trees_ros.trees.BehaviourTree so py-trees-tree-viewer can discover/open streams.
"""

import rclpy
from rclpy.node import Node
import py_trees
from py_trees.visitors import SnapshotVisitor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy  # (kept for future use)
from py_trees_ros.trees import BehaviourTree as RosBehaviourTree

from behavior_controller.behavior_controller_tree import create_tour_guide_tree, display_tree


class TourGuideNode(Node):
    """
    ROS 2 node that manages the tour guide behavior tree with ROS-integrated snapshot services.
    """

    def __init__(self):
        super().__init__('behavior_controller_node')  # keep same name you’re using on topics

        # -------------------- Parameters --------------------
        self.declare_parameter('bt_loop_rate', 10)           # Hz
        self.declare_parameter('display_tree_on_start', True)
        self.declare_parameter('show_live_status', False)

        self.bt_loop_rate = self.get_parameter('bt_loop_rate').value
        self.display_on_start = self.get_parameter('display_tree_on_start').value
        self.show_live_status = self.get_parameter('show_live_status').value

        self.get_logger().info("=" * 80)
        self.get_logger().info("Tour Guide BehaviorTree Node Starting...")
        self.get_logger().info(f"  Tick rate: {self.bt_loop_rate} Hz")
        self.get_logger().info("=" * 80)

        # -------------------- Build & Setup Tree --------------------
        try:
            root = create_tour_guide_tree(self)

            self.tree = RosBehaviourTree(root)
            
            # Pass the rclpy node to setup so it can create services/pubs internally
            self.tree.setup(node=self, timeout=15.0)

            # Optional visitor for live console snapshot (and other analytics)
            self.snapshot_visitor = SnapshotVisitor()
            self.tree.visitors.append(self.snapshot_visitor)

            # Enable the *default* snapshot stream (the viewer knows how to find this)
            # These are runtime node params consumed by py_trees_ros.trees.BehaviourTree.
            # After this, the viewer can attach to /tree/snapshots automatically.
            self.set_parameters([
                rclpy.parameter.Parameter(
                    'default_snapshot_stream',
                    rclpy.Parameter.Type.BOOL, True
                ),
                rclpy.parameter.Parameter(
                    'default_snapshot_blackboard_data',
                    rclpy.Parameter.Type.BOOL, True
                ),
                rclpy.parameter.Parameter(
                    'default_snapshot_blackboard_activity',
                    rclpy.Parameter.Type.BOOL, True
                ),
                rclpy.parameter.Parameter(
                    'default_snapshot_period',
                    rclpy.Parameter.Type.DOUBLE, 0.5  # seconds between updates for default stream
                ),
            ])

            if self.display_on_start:
                display_tree(root)

            self.get_logger().info("BehaviorTree loaded and setup successfully!")
            self.get_logger().info("Viewer will connect via the tree’s snapshot stream services.")
            self.get_logger().info("Tip: run `py-trees-tree-viewer` (no args) and pick your tree if prompted.")

        except Exception as e:
            self.get_logger().error(f"Error creating BehaviorTree: {e}")
            import traceback
            traceback.print_exc()
            raise

        # -------------------- Tick Timer --------------------
        self.previous_status = None
        self.tour_count = 0
        self.tick_count = 0
        timer_period = 1.0 / float(self.bt_loop_rate)
        self.timer = self.create_timer(timer_period, self.tick_tree)

        self.get_logger().info(f"Tour Guide Node ready! Ticking at {self.bt_loop_rate} Hz")

    # -------------------- Tick & Handlers --------------------
    def tick_tree(self):
        try:
            self.tree.tick()
            self.tick_count += 1

            if self.show_live_status:
                self.display_tree_status()

            status = self.tree.root.status
            if status != self.previous_status:
                self.log_status_change(status)
                self.previous_status = status

            if status == py_trees.common.Status.SUCCESS:
                self.handle_tour_success()
            elif status == py_trees.common.Status.FAILURE:
                self.handle_tour_failure()

        except Exception as e:
            self.get_logger().error(f"Error ticking tree: {e}")

    def log_status_change(self, status):
        if status == py_trees.common.Status.RUNNING:
            self.get_logger().debug("Tree status: RUNNING")
        elif status == py_trees.common.Status.SUCCESS:
            self.get_logger().info("✅ Tree status: SUCCESS")
        elif status == py_trees.common.Status.FAILURE:
            self.get_logger().warn("❌ Tree status: FAILURE")
        elif status == py_trees.common.Status.INVALID:
            self.get_logger().warn("Tree status: INVALID")

    def handle_tour_success(self):
        self.tour_count += 1
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"🎉 Tour #{self.tour_count} completed successfully!")
        self.get_logger().info("=" * 80)
        # Reset for next tour
        self.tree.root.stop(py_trees.common.Status.INVALID)
        self.previous_status = None
        self.get_logger().info("Ready for next tour...")

    def handle_tour_failure(self):
        self.get_logger().info("=" * 80)
        self.get_logger().info("Tour failed or was cancelled")
        self.get_logger().info("=" * 80)
        # Reset for next tour
        self.tree.root.stop(py_trees.common.Status.INVALID)
        self.previous_status = None
        self.get_logger().info("Ready for next tour...")

    def display_tree_status(self):
        # Clear screen and print a compact live tree view of only visited nodes
        print("\033[2J\033[H", end='')
        print("\n" + "=" * 80)
        print("LIVE TREE STATUS")
        print("=" * 80)
        print(py_trees.display.unicode_tree(
            self.tree.root,
            show_status=True,
            show_only_visited=True
        ))
        print("=" * 80)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TourGuideNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nShutting down tour guide node...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
