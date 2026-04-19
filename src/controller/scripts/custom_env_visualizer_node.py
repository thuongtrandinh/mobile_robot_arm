#!/usr/bin/env python3

import rclpy

from ampcc_rviz_visualizer import AmpccRvizVisualizer


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AmpccRvizVisualizer()
    node.get_logger().warn(
        "custom_env_visualizer_node.py is deprecated. Merged visualizer now runs in ampcc_rviz_visualizer.py"
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
