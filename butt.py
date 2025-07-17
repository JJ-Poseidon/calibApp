import tkinter as tk
import math

class BoatSimulation(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Boat Navigation Simulator")
        self.geometry("800x600")
        self.configure(bg="#a0c8f0")  # soft blue ocean color

        self.canvas = tk.Canvas(self, width=800, height=600, bg="#a0c8f0", highlightthickness=0)
        self.canvas.pack()

        # Boat state
        self.boat_pos = [400, 300]  # center start
        self.boat_angle = 0  # facing right (degrees)
        self.boat_velocity = [0.0, 0.0]
        self.boat_speed = 0.0  # scalar speed
        self.boat_acceleration = 0.0  # scalar acceleration (positive or negative)
        self.max_speed = 5.0
        self.min_speed = 0.0

        # Turning dynamics
        self.boat_angular_velocity = 0.0  # degrees/frame
        self.boat_angular_acceleration = 0.0
        self.max_angular_speed = 3.0  # max degrees per frame
        self.turn_acceleration = 0.1  # how fast it turns

        # Tide (current) vector
        self.tide_speed = 1.0
        self.tide_direction = 90  # degrees (0 = right, 90 = down)

        # Waypoints list: list of (x,y)
        self.waypoints = []
        self.current_waypoint_index = 0
        self.waypoint_radius = 15

        # Controls for tide
        control_frame = tk.Frame(self)
        control_frame.pack()

        tk.Label(control_frame, text="Tide Speed").grid(row=0, column=0)
        self.tide_speed_slider = tk.Scale(control_frame, from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_tide_speed)
        self.tide_speed_slider.set(self.tide_speed)
        self.tide_speed_slider.grid(row=0, column=1)

        tk.Label(control_frame, text="Tide Direction").grid(row=1, column=0)
        self.tide_dir_slider = tk.Scale(control_frame, from_=0, to=360, orient=tk.HORIZONTAL, command=self.update_tide_direction)
        self.tide_dir_slider.set(self.tide_direction)
        self.tide_dir_slider.grid(row=1, column=1)

        # Bind mouse click to set waypoints
        self.canvas.bind("<Button-1>", self.add_waypoint)

        # Input state for acceleration and turning
        self.bind("<KeyPress-w>", self.accelerate)
        self.bind("<KeyRelease-w>", self.stop_accelerate)
        self.bind("<KeyPress-s>", self.brake)
        self.bind("<KeyRelease-s>", self.stop_brake)
        self.bind("<KeyPress-a>", self.turn_left)
        self.bind("<KeyPress-d>", self.turn_right)
        self.bind("<KeyRelease-a>", self.stop_turn)
        self.bind("<KeyRelease-d>", self.stop_turn)

        self.is_accelerating = False
        self.is_braking = False
        self.turning_direction = 0  # -1 for left, +1 for right, 0 no turn

        self.boat_shape = None

        # Start animation
        self.after(30, self.update_simulation)

    def update_tide_speed(self, val):
        self.tide_speed = float(val)

    def update_tide_direction(self, val):
        self.tide_direction = float(val)

    def add_waypoint(self, event):
        self.waypoints.append((event.x, event.y))
        self.draw_waypoints()

    def draw_waypoints(self):
        self.canvas.delete("waypoint")
        for (x, y) in self.waypoints:
            self.canvas.create_oval(x-7, y-7, x+7, y+7, fill="red", tags="waypoint")

    def accelerate(self, event):
        self.is_accelerating = True

    def stop_accelerate(self, event):
        self.is_accelerating = False

    def brake(self, event):
        self.is_braking = True

    def stop_brake(self, event):
        self.is_braking = False

    def turn_left(self, event):
        self.turning_direction = -1

    def turn_right(self, event):
        self.turning_direction = 1

    def stop_turn(self, event):
        # Only stop turning if released key matches current direction
        # (to handle holding both keys weirdness)
        self.turning_direction = 0

    def draw_grid(self):
        self.canvas.delete("grid")
        spacing = 50
        width = int(self.canvas['width'])
        height = int(self.canvas['height'])

        for x in range(0, width, spacing):
            self.canvas.create_line(x, 0, x, height, fill="#d0e4ff", tags="grid")
        for y in range(0, height, spacing):
            self.canvas.create_line(0, y, width, y, fill="#d0e4ff", tags="grid")

    def update_simulation(self):
        self.draw_grid()

        # Update acceleration based on input
        accel_rate = 0.05
        decel_rate = 0.02

        if self.is_accelerating:
            self.boat_acceleration = accel_rate
        elif self.is_braking:
            self.boat_acceleration = -accel_rate * 2  # stronger braking
        else:
            # Natural friction slows the boat
            if self.boat_speed > 0:
                self.boat_acceleration = -decel_rate
            else:
                self.boat_acceleration = 0

        # Update speed with acceleration
        self.boat_speed += self.boat_acceleration
        self.boat_speed = max(self.min_speed, min(self.max_speed, self.boat_speed))

        # Update angular acceleration and velocity
        if self.turning_direction != 0:
            self.boat_angular_acceleration = self.turn_acceleration * self.turning_direction
        else:
            # Slow angular velocity gradually (simulate rotational friction)
            self.boat_angular_acceleration = -self.boat_angular_velocity * 0.1

        self.boat_angular_velocity += self.boat_angular_acceleration
        # Clamp angular velocity
        if self.boat_angular_velocity > self.max_angular_speed:
            self.boat_angular_velocity = self.max_angular_speed
        elif self.boat_angular_velocity < -self.max_angular_speed:
            self.boat_angular_velocity = -self.max_angular_speed

        # Update boat angle
        self.boat_angle += self.boat_angular_velocity
        self.boat_angle %= 360

        # Update velocity vector based on speed and angle
        angle_rad = math.radians(self.boat_angle)
        self.boat_velocity = [self.boat_speed * math.cos(angle_rad),
                              self.boat_speed * math.sin(angle_rad)]

        # Add tide current
        tide_rad = math.radians(self.tide_direction)
        tide_vector = [self.tide_speed * math.cos(tide_rad),
                       self.tide_speed * math.sin(tide_rad)]

        # Total velocity including tide
        total_velocity = [self.boat_velocity[0] + tide_vector[0],
                          self.boat_velocity[1] + tide_vector[1]]

        # Update boat position
        self.boat_pos[0] += total_velocity[0]
        self.boat_pos[1] += total_velocity[1]

        # Waypoint navigation logic
        if self.waypoints:
            target = self.waypoints[self.current_waypoint_index]
            dx = target[0] - self.boat_pos[0]
            dy = target[1] - self.boat_pos[1]
            distance = math.hypot(dx, dy)

            if distance < self.waypoint_radius:
                self.current_waypoint_index += 1
                if self.current_waypoint_index >= len(self.waypoints):
                    self.current_waypoint_index = 0  # loop back

            # Steering toward waypoint (override turning direction)
            desired_angle = math.degrees(math.atan2(dy, dx)) % 360
            angle_diff = (desired_angle - self.boat_angle + 540) % 360 - 180  # shortest angle difference (-180 to 180)

            # Control angular velocity to reduce angle difference smoothly
            turn_sensitivity = 0.05
            self.boat_angular_velocity += turn_sensitivity * angle_diff
            # Clamp angular velocity again after correction
            if self.boat_angular_velocity > self.max_angular_speed:
                self.boat_angular_velocity = self.max_angular_speed
            elif self.boat_angular_velocity < -self.max_angular_speed:
                self.boat_angular_velocity = -self.max_angular_speed

            # Accelerate if roughly facing waypoint
            if abs(angle_diff) < 30:
                self.is_accelerating = True
                self.is_braking = False
            else:
                self.is_accelerating = False

        self.draw_boat()
        self.draw_speed()

        self.after(30, self.update_simulation)

    def draw_boat(self):
        if self.boat_shape:
            self.canvas.delete(self.boat_shape)

        size = 25
        x, y = self.boat_pos

        # Rotate the boat shape by -90 degrees so bow points forward at boat_angle
        angle_rad = math.radians(self.boat_angle + 90)

        points = [
            (0, -size),          # bow (front tip)
            (size * 0.6, -size * 0.3),  # front right
            (size * 0.6, size * 0.3),   # rear right
            (0, size * 0.5),     # stern middle bottom
            (-size * 0.6, size * 0.3),  # rear left
            (-size * 0.6, -size * 0.3)  # front left
        ]

        rotated_points = []
        for px, py in points:
            rx = px * math.cos(angle_rad) - py * math.sin(angle_rad)
            ry = px * math.sin(angle_rad) + py * math.cos(angle_rad)
            rotated_points.append((x + rx, y + ry))

        flat_points = [coord for point in rotated_points for coord in point]

        self.boat_shape = self.canvas.create_polygon(
            flat_points,
            fill="white",
            outline="black",
            width=2
        )


    def draw_speed(self):
        speed = math.hypot(*self.boat_velocity)
        self.canvas.delete("speed_text")
        self.canvas.create_text(10, 10, anchor="nw",
                                text=f"Speed: {speed:.2f} px/frame",
                                fill="black", font=("Arial", 14, "bold"),
                                tags="speed_text")

if __name__ == "__main__":
    app = BoatSimulation()
    app.mainloop()
