import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from cssr_interfaces.action import TTS
from .implementation import TTSProcessor

class RobotTTSActionServer(Node):
    def __init__(self):
        super().__init__('robot_tts_action_server')
        self.processor = TTSProcessor(self)
        
        self._action_server = ActionServer(
            self, TTS, 'tts_action',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        self.get_logger().info("Robot TTS Server Started")

    def goal_callback(self, goal_request):
        return GoalResponse.ACCEPT if goal_request.text.strip() else GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        self.processor.stream.stop()
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        feedback_msg = TTS.Feedback()

        def update_feedback(status_text):
            feedback_msg.status = status_text
            goal_handle.publish_feedback(feedback_msg)

        success, message = self.processor.process_and_stream(
            goal_handle.request.text, update_feedback
        )

        result = TTS.Result()
        result.success = success
        result.message = message
        
        if success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

def main(args=None):
    rclpy.init(args=args)
    node = RobotTTSActionServer()
    # MultiThreadedExecutor is crucial for concurrent feedback/cancellation
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()