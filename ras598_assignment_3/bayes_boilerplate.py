import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from marker_msgs.msg import MarkerDetection
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from builtin_interfaces.msg import Time
import numpy as np
from scipy.ndimage import gaussian_filter
def euler_from_quaternion(q):
    """Return (roll, pitch, yaw) from quaternion [x, y, z, w]."""
    x, y, z, w = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw
import re
import os

class BayesFilter3D(Node):
    def __init__(self, world_file_path):
        super().__init__('bayes_filter_3d_node')
        
        # --- CONFIGURATION ---
        self.world_size = 16.0         # Total width/height of the environment (meters)
        self.resolution = 0.2          # Size of one grid cell (meters)
        self.theta_res = 5             # Angular resolution (degrees)
        self.grid_dim = int(self.world_size / self.resolution)
        self.theta_dim = int(360 / self.theta_res)

        # --- ROS PUBLISHERS ---
        self.costmap_pub = self.create_publisher(OccupancyGrid, 'viz/belief_costmap', 10)
        self.landmark_pub = self.create_publisher(MarkerArray, 'viz/landmarks', 10)
        self.gt_path_pub = self.create_publisher(Path, 'viz/gt_path', 10)
        self.odom_path_pub = self.create_publisher(Path, 'viz/odom_path', 10)
        
        # Path messages initialization
        self.gt_path_msg = Path(header=Header(frame_id='map'))
        self.odom_path_msg = Path(header=Header(frame_id='map'))

        # --- FILTER STATE INITIALIZATION ---
        self.landmarks = self._parse_world_file(world_file_path)
        self.initial_pose = [-7.0, -7.0, 90.0]  # [x, y, theta_degrees] DO NOT CHANGE THIS!
        
        # Trajectory tracking for visualization
        self.odom_x, self.odom_y = self.initial_pose[0], self.initial_pose[1]
        self.odom_th = np.radians(self.initial_pose[2])
        
        # Initialize self.belief in initialize_belief()
        self.initialize_belief(pose=self.initial_pose)

        self.last_odom_pose = None

        # --- SUBSCRIPTIONS ---
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Odometry, '/ground_truth', self.gt_callback, 10)
        self.create_subscription(MarkerDetection, '/fiducials', self._fiducial_callback, 10)

        # Refresh landmarks and belief in RViz
        self.create_timer(1.0, self._publish_landmarks)
        self.create_timer(1.0, self._publish_costmap)

        # Print Landmarks Locations
        self.get_logger().info("--- Landmark Locations ---")
        for tid, pos in self.landmarks.items():
            lx, ly = pos
            self.get_logger().info(f"ID {tid}: x={lx:.2f}, y={ly:.2f}")
        self.get_logger().info("---------------------------")
    
    # -------------------------------------------------------------------------
    # UTILITY & VISUALIZATION FUNCTIONS
    # You don't need to change the function code for these functions.
    # -------------------------------------------------------------------------

    def _parse_world_file(self, path):
        """Parses Stage .world file for landmark positions."""
        found = {}
        if not os.path.exists(path): return found
        with open(path, 'r') as f:
            content = f.read()
        block_pattern = re.compile(r'my_block\s*\((.*?)\)', re.DOTALL)
        pose_pattern = re.compile(r'pose\s*\[\s*([-\d.]+)\s+([-\d.]+)')
        id_pattern = re.compile(r'fiducial_return\s+(\d+)')
        for block_content in block_pattern.findall(content):
            p_match = pose_pattern.search(block_content)
            id_match = id_pattern.search(block_content)
            if p_match and id_match:
                found[int(id_match.group(1))] = (float(p_match.group(1)), float(p_match.group(2)))
        return found

    def _publish_landmarks(self):
        """Publishes landmark locations as Markers for RViz."""
        ma = MarkerArray()
        for tid, (tx, ty) in self.landmarks.items():
            # Cylinder Marker
            c = Marker(header=Header(frame_id='map'), id=tid, type=Marker.CYLINDER, action=Marker.ADD)
            c.pose.position.x, c.pose.position.y, c.pose.position.z = tx, ty, 0.5
            c.scale.x, c.scale.y, c.scale.z = 0.3, 0.3, 1.0
            c.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            ma.markers.append(c)
            # Text ID Marker
            t = Marker(header=Header(frame_id='map'), id=tid + 1000, type=Marker.TEXT_VIEW_FACING)
            t.text = f"ID: {tid}"
            t.pose.position.x, t.pose.position.y, t.pose.position.z = tx, ty + 0.5, 1.2
            t.scale.z = 0.4 
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            ma.markers.append(t)
        self.landmark_pub.publish(ma)

    def _publish_costmap(self):
        """Publishes the 2D projected belief as an OccupancyGrid."""
        if not hasattr(self, 'belief'): return
        grid = OccupancyGrid(header=Header(frame_id='map', stamp=self.get_clock().now().to_msg()))
        grid.info.resolution, grid.info.width, grid.info.height = self.resolution, self.grid_dim, self.grid_dim
        grid.info.origin.position.x, grid.info.origin.position.y = -8.0, -8.0
        grid.info.origin.orientation.w = 1.0

        belief_2d = np.sum(self.belief, axis=2)
        belief_flipped = np.flipud(belief_2d) # Match ROS bottom-up convention
        
        max_val = np.max(belief_flipped)
        if max_val > 0:
            data = (belief_flipped / max_val * 100).astype(np.int8)
        else:
            data = np.zeros_like(belief_flipped, dtype=np.int8)
        
        grid.data = data.flatten().tolist()
        self.costmap_pub.publish(grid)
    
    
    # -------------------------------------------------------------------------
    #  ASSIGNMENT TASKS
    # -------------------------------------------------------------------------

    def gt_callback(self, msg):
        """Updates Ground Truth path for visualization."""
        p = PoseStamped(header=Header(frame_id='map', stamp=msg.header.stamp))
        p.pose.position.x, p.pose.position.y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self.gt_path_msg.poses.append(p)
        self.gt_path_pub.publish(self.gt_path_msg)

    def initialize_belief(self, pose=None):
        """
        TASK 1: Initialize the 3D Probability Density Function (PDF).
        If 'pose' is provided, initialize the belief as a localized
        distribution (e.g., a point or Gaussian). If 'pose' is None,
        initialize a Uniform distribution across the entire state space.
        """
        self.belief = np.zeros((self.grid_dim, self.grid_dim, self.theta_dim))
        if pose is None:
            self.belief[:] = 1.0
        else:
            ix, iy, ith = self.real_to_grid(pose[0], pose[1], pose[2])
            self.belief[iy, ix, ith] = 1.0
            self.belief = gaussian_filter(self.belief, sigma=[2.0, 2.0, 1.0])
        self.belief /= np.sum(self.belief)

    def real_to_grid(self, x, y, theta_deg):
        """
        TASK 2(a): Coordinate Transformation (Real -> Grid).
        Convert continuous world coordinates to discrete 3D grid indices.
        Returns: (ix, iy, ith)
        """
        ix  = int((x + 8.0) / self.resolution)
        iy  = int((8.0 - y) / self.resolution)   # iy=0 = top = highest y
        ith = int(theta_deg % 360 / self.theta_res) % self.theta_dim
        ix  = int(np.clip(ix, 0, self.grid_dim - 1))
        iy  = int(np.clip(iy, 0, self.grid_dim - 1))
        return ix, iy, ith

    def grid_to_real(self):
        """
        TASK 2(b): Coordinate Transformation (Grid -> Real).
        Generate 3D numpy arrays containing the real-world (x, y, theta)
        values for every cell in the belief grid.
        Returns: rx, ry, rth (all numpy arrays of shape self.belief.shape)
        """
        ix_arr = np.arange(self.grid_dim)   # varies along axis 1 (x / columns)
        iy_arr = np.arange(self.grid_dim)   # varies along axis 0 (y / rows)
        ith_arr = np.arange(self.theta_dim) # varies along axis 2 (theta)

        rx  = (ix_arr[np.newaxis, :, np.newaxis] + 0.5) * self.resolution - 8.0
        ry  = 8.0 - (iy_arr[:, np.newaxis, np.newaxis] + 0.5) * self.resolution
        rth = ith_arr[np.newaxis, np.newaxis, :] * float(self.theta_res)

        # Broadcast to full belief shape (grid_dim, grid_dim, theta_dim)
        rx  = np.broadcast_to(rx,  self.belief.shape)
        ry  = np.broadcast_to(ry,  self.belief.shape)
        rth = np.broadcast_to(rth, self.belief.shape)
        return rx, ry, rth

    def predict(self, curr_msg, last_msg):
        """
        TASK 3: Motion Model (Prediction).
        Implement the 'Turn-Go-Turn' model. Update self.belief to reflect
        the robot's movement between last_msg and curr_msg.
        """
        # Extract yaw angles from quaternions
        q = curr_msg.pose.pose.orientation
        curr_yaw = np.degrees(euler_from_quaternion([q.x, q.y, q.z, q.w])[2]) % 360
        q_old = last_msg.pose.pose.orientation
        old_yaw = np.degrees(euler_from_quaternion([q_old.x, q_old.y, q_old.z, q_old.w])[2]) % 360

        dx = curr_msg.pose.pose.position.x - last_msg.pose.pose.position.x
        dy = curr_msg.pose.pose.position.y - last_msg.pose.pose.position.y
        dth = (curr_yaw - old_yaw + 180) % 360 - 180   # signed, degrees

        d_trans = np.sqrt(dx**2 + dy**2)

        # Turn-Go-Turn decomposition
        if d_trans > 1e-6:
            move_dir = np.degrees(np.arctan2(dy, dx)) % 360
            d_rot1 = (move_dir - old_yaw + 180) % 360 - 180
        else:
            d_rot1 = 0.0
        d_rot2 = dth - d_rot1

        # Step 1: apply d_rot1 — shift belief along the theta axis
        shift_rot1 = int(round(d_rot1 / self.theta_res)) % self.theta_dim
        self.belief = np.roll(self.belief, shift_rot1, axis=2)

        # Step 2: apply translation — each theta-slice shifts by a different (dx, dy)
        new_belief = np.zeros_like(self.belief)
        for ith in range(self.theta_dim):
            theta_rad = np.radians(ith * self.theta_res)
            shift_x = int(round(d_trans * np.cos(theta_rad) / self.resolution))
            shift_y = int(round(d_trans * np.sin(theta_rad) / self.resolution))
            new_belief[:, :, ith] = np.roll(
                np.roll(self.belief[:, :, ith], -shift_y, axis=0),
                shift_x, axis=1
            )
        self.belief = new_belief

        # Step 3: apply d_rot2
        shift_rot2 = int(round(d_rot2 / self.theta_res)) % self.theta_dim
        self.belief = np.roll(self.belief, shift_rot2, axis=2)

        # Step 4: Gaussian diffusion to spread uncertainty
        self.belief = gaussian_filter(self.belief, sigma=[0.5, 0.5, 0.3])

        # Normalize
        total = np.sum(self.belief)
        if total > 0:
            self.belief /= total

        self.belief = 0.95 * self.belief + 0.05 / self.belief.size
        self.belief /= np.sum(self.belief)

    def update_measurement(self, measured_range, measured_bearing_deg, landmark_pos):
        """
        TASK 4: Measurement Model (Update).
        Correct the belief using a landmark sighting.
        """
        rx, ry, rth = self.grid_to_real()
        lx, ly = landmark_pos

        dx = lx - rx
        dy = ly - ry

        # Expected range and bearing from every grid cell to the landmark
        exp_range   = np.sqrt(dx**2 + dy**2)
        exp_bearing = (np.degrees(np.arctan2(dy, dx)) - rth + 180) % 360 - 180  # [-180, 180]

        # Normalize measured bearing to same range
        meas_bearing = (measured_bearing_deg + 180) % 360 - 180

        # Gaussian likelihood — noise params match Stage sensor (range_error=0.5, bearing_error~11.5 deg)
        sigma_r = 0.5
        sigma_b = 15.0
        likelihood = (
            np.exp(-0.5 * ((measured_range - exp_range) / sigma_r) ** 2) *
            np.exp(-0.5 * ((meas_bearing   - exp_bearing) / sigma_b) ** 2)
        )

        self.belief *= likelihood
        total = np.sum(self.belief)
        if total > 1e-300:
            self.belief /= total
        else:
            # Degenerate: reset to uniform to avoid NaN propagation
            self.belief[:] = 1.0 / self.belief.size


    def odom_callback(self, msg):
        """Handles robot motion, trajectory visualization and performas the Prediction update."""
        if self.last_odom_pose is None:
            self.last_odom_pose = msg
            return
        
        q = msg.pose.pose.orientation
        curr_yaw_deg = np.degrees(euler_from_quaternion([q.x, q.y, q.z, q.w])[2]) % 360
        
        q_old = self.last_odom_pose.pose.pose.orientation
        old_yaw_deg = np.degrees(euler_from_quaternion([q_old.x, q_old.y, q_old.z, q_old.w])[2]) % 360

        # Calculates the differential odometry
        dx = msg.pose.pose.position.x - self.last_odom_pose.pose.pose.position.x
        dy = msg.pose.pose.position.y - self.last_odom_pose.pose.pose.position.y
        dth = (curr_yaw_deg - old_yaw_deg + 180) % 360 - 180 

        # Update and Publish Odom Path
        self.odom_th += np.radians(dth)
        self.odom_x += (dx * np.cos(self.odom_th)) - (dy * np.sin(self.odom_th))
        self.odom_y += (dx * np.sin(self.odom_th)) + (dy * np.cos(self.odom_th))
        
        p = PoseStamped(header=Header(frame_id='map', stamp=msg.header.stamp))
        p.pose.position.x, p.pose.position.y = float(self.odom_x), float(self.odom_y)
        self.odom_path_msg.poses.append(p)
        self.odom_path_pub.publish(self.odom_path_msg)

        # Run the prediction loop only when there is sufficient motion
        if np.sqrt(dx**2 + dy**2) > 0.001 or abs(dth) > 0.1:
            self.predict(msg, self.last_odom_pose)
            self.last_odom_pose = msg
            self._publish_costmap()
    
    def _fiducial_callback(self, msg):
        """
        TASK 5: Fiducial Callback.
        Performs the Measurement update for every landmark (also called marker) seen by the robot.
        """
        for marker in msg.markers:
            if not marker.ids:
                continue
            tid = marker.ids[0]
            if tid not in self.landmarks:
                continue
            # Marker pose is in the robot's sensor frame: x=forward, y=left
            mx = marker.pose.position.x
            my = marker.pose.position.y
            measured_range   = np.sqrt(mx**2 + my**2)
            measured_bearing = np.degrees(np.arctan2(my, mx))
            self.update_measurement(measured_range, measured_bearing, self.landmarks[tid])

        # Publishes the probability distribution costmap
        self._publish_costmap()  # Dont remove this line
    

def main():
    rclpy.init()

    world_path = os.path.expanduser("~/ros2_ws/src/stage_ros2/world/cave.world")
    if not os.path.exists(world_path):
        print("\n" + "="*50)
        print("ERROR: World file not found!")
        print(f"Path attempted: {world_path}")
        print("-" * 50)
        print("FIX: Please open your script and update the 'world_path' variable")
        print("to match your ROS 2 workspace name (e.g., ~/dev_ws/src/...)")
        print("="*50 + "\n")
        return # Exit the program gracefully


    node = BayesFilter3D(world_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()