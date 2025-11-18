import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator

def main():
    rclpy.init()
    navigator = BasicNavigator()
    
    # Set initial pose
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = navigator.get_clock().now().to_msg()
    initial_pose.pose.position.x = 0.0
    initial_pose.pose.position.y = 0.0
    initial_pose.pose.orientation.w = 1.0
    navigator.setInitialPose(initial_pose)
    
    # Wait for Nav2 to activate
    navigator.waitUntilNav2Active()
    
    # Send goal
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = navigator.get_clock().now().to_msg()
    goal_pose.pose.position.x = 2.0
    goal_pose.pose.position.y = 1.0
    goal_pose.pose.orientation.w = 1.0
    
    navigator.goToPose(goal_pose)
    
    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        print(f'Distance remaining: {feedback.distance_remaining}')
    
    print('Goal reached!')
    rclpy.shutdown()

if __name__ == '__main__':
    main()