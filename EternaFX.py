import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
from matplotlib.widgets import Slider  # Import Slider widget
from matplotlib import cm  # Import colormaps


class EternaFX:
    """
    EternaFX Framework for 4D to 3D projections and animations.
    Enhanced with visual effects and basic interactivity for a more engaging experience.
    """

    def __init__(self):
        """Initializes EternaFX with enhanced visual settings and interactive elements."""
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(8, 8)) # Create figure and axes using subplots for widget placement
        self.scatter = self.ax.scatter([], [], [], c=[], s=50, marker='o', alpha=0.8)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('EternaFX: Interactive 4D Projection Animation') # Updated title for interactivity
        self.color_data = None
        self.point_size = 50
        self.rotation_speed_multiplier = 1.0 # Initial rotation speed multiplier
        self.colormap = cm.viridis # Default colormap


    def generate_4d_hypersphere(self, num_points=100):
        """Generates points on a 4D hypersphere."""
        t = np.linspace(0, 2 * np.pi, num_points)
        x = np.sin(t)
        y = np.cos(t)
        z = np.sin(2 * t)
        w = np.cos(2 * t)
        self.color_data = t
        return np.vstack([x, y, z, w]).T

    def generate_4d_hypercube(self, num_points_per_edge=10):
        """Generates points for a 4D hypercube."""
        points_1d = np.linspace(-1, 1, num_points_per_edge)
        points_4d = []
        for x in points_1d:
            for y in points_1d:
                for z in points_1d:
                    for w in points_1d:
                        points_4d.append([x, y, z, w])
        self.color_data = np.arctan2(np.array(points_4d)[:, 0], np.array(points_4d)[:, 1])
        return np.array(points_4d)

    def transform_4d_cube(self, points_4d, frame_num, transformation_type='shear_w'):
        """Applies transformations to the 4D hypercube in the 4th dimension."""
        transformed_points = np.copy(points_4d)
        if transformation_type == 'shear_w':
            shear_factor = np.sin(frame_num / 20.0) * 1.5
            transformed_points[:, 3] += shear_factor * transformed_points[:, 0]
        elif transformation_type == 'scale_w':
            scale_factor_w = 1 + np.sin(frame_num / 30.0) * 0.5
            transformed_points[:, 3] *= scale_factor_w
        return transformed_points


    def project_4d_to_3d(self, points_4d, frame_num, rotation_axes='xyz', rotation_angles=[45, 30, 15], rotation_speeds=[1, 0.5, 0.8], scale_factor=1.0): # Default rotation speeds adjusted
        """Projects 4D points to 3D with rotation and scaling."""
        angles = []
        if 'x' in rotation_axes:
            angles.append(rotation_angles[0] + frame_num * rotation_speeds[0] * self.rotation_speed_multiplier) # Apply speed multiplier
        else:
            angles.append(0)
        if 'y' in rotation_axes:
            angles.append(rotation_angles[1] + frame_num * rotation_speeds[1] * self.rotation_speed_multiplier) # Apply speed multiplier
        else:
            angles.append(0)
        if 'z' in rotation_axes:
            angles.append(rotation_angles[2] + frame_num * rotation_speeds[2] * self.rotation_speed_multiplier) # Apply speed multiplier
        else:
            if len(rotation_angles) < 3:
                angles.append(0)
            else:
                angles.append(0)

        rot = R.from_euler(rotation_axes, angles[:len(rotation_axes)], degrees=True)
        points_3d_rotated = rot.apply(points_4d[:, :3])
        return points_3d_rotated * scale_factor

    def adjust_axes_limits(self, points_3d, padding_factor=1.2):
        """Adjusts 3D axes limits to tightly fit the data."""
        x_range = np.ptp(points_3d[:, 0])
        y_range = np.ptp(points_3d[:, 1])
        z_range = np.ptp(points_3d[:, 2])
        max_range = max(x_range, y_range, z_range) / 2.0
        mid_x = (np.max(points_3d[:, 0]) + np.min(points_3d[:, 0])) * 0.5
        mid_y = (np.max(points_3d[:, 1]) + np.min(points_3d[:, 1])) * 0.5
        mid_z = (np.max(points_3d[:, 2]) + np.min(points_3d[:, 2])) * 0.5
        self.ax.set_xlim(mid_x - max_range * padding_factor, mid_x + max_range * padding_factor)
        self.ax.set_ylim(mid_y - max_range * padding_factor, mid_y + max_range * padding_factor)
        self.ax.set_zlim(mid_z - max_range * padding_factor, mid_z + max_range * padding_factor)


    def animate(self, frame_num, points_4d, animation_settings):
        """Animation function with color cycling and dynamic updates."""
        shape_type = animation_settings.get('shape_type', 'hypersphere')
        transformation_type_4d = animation_settings.get('transformation_type_4d', None)
        rotation_axes = animation_settings.get('rotation_axes', 'xyz')
        rotation_angles = animation_settings.get('rotation_angles', [45, 30, 15])
        rotation_speeds = animation_settings.get('rotation_speeds', [1, 0.5, 0.8]) # Adjusted default speeds
        scale_factor = animation_settings.get('scale_factor', 1.0)
        camera_elevation = animation_settings.get('camera_elevation', 30)
        camera_azimuth = animation_settings.get('camera_azimuth', 30)
        color_cycle_speed = animation_settings.get('color_cycle_speed', 10) # Color cycle speed setting

        if transformation_type_4d and shape_type == 'hypercube':
            points_4d = self.transform_4d_cube(points_4d, frame_num, transformation_type_4d)

        points_3d = self.project_4d_to_3d(points_4d, frame_num, rotation_axes, rotation_angles, rotation_speeds, scale_factor)

        self.scatter._offsets3d = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])

        # Color Cycling Effect
        color_indices = (self.color_data + frame_num / color_cycle_speed) % (2 * np.pi) # Cycle colors based on frame
        self.scatter.set_array(color_indices) # Update scatter plot colors
        self.scatter.set_cmap(self.colormap) # Apply colormap


        self.adjust_axes_limits(points_3d)
        self.ax.view_init(elev=camera_elevation, azim=camera_azimuth + frame_num * 0.5)
        self.ax.set_title(f'EternaFX: Frame {frame_num} - Interactive 4D Animation') # Updated title
        return self.scatter,

    def run_animation(self, points_4d, animation_settings, frames=200, interval=50, save_animation=False, filename='eternafx_animation.mp4'):
        """Runs animation and sets up interactive slider."""
        ani = animation.FuncAnimation(self.fig, self.animate, fargs=(points_4d, animation_settings),
                                      frames=frames, interval=interval, blit=False)

        # --- Rotation Speed Slider ---
        ax_slider = plt.axes([0.25, 0.01, 0.50, 0.03]) # Position of slider
        speed_slider = Slider(ax=ax_slider, label='Rotation Speed', valmin=0.1, valmax=3.0, valinit=1.0) # Slider properties

        def update_speed(val): # Function to update rotation speed
            self.rotation_speed_multiplier = val

        speed_slider.on_changed(update_speed) # Connect slider to update function


        if save_animation:
            try:
                ani.save(filename, writer='ffmpeg')
                print(f"Animation saved as {filename} (MP4 format).")
            except Exception as e:
                print(f"Error saving MP4 animation. Ensure ffmpeg is installed. Trying GIF.\nError: {e}")
                try:
                    ani.save(filename.replace('.mp4', '.gif'), writer='pillow')
                    print(f"Animation saved as {filename.replace('.mp4', '.gif')} (GIF format).")
                except Exception as e_gif:
                    print(f"Error saving GIF animation. Check writer installations.\nGIF Error: {e_gif}")

        plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for slider
        plt.show()


if __name__ == '__main__':
    efx = EternaFX()

    print("EternaFX Interactive Framework Demonstration")
    print("--------------------------------------------\n")
    print("Explore interactive 4D visualizations! Use the slider to control rotation speed.")
    print("Customize shapes, transformations, rotation, scaling, camera, and color cycling.\n")


    # --- Interactive Hypersphere Animation with Color Cycling ---
    print("Example 1: Interactive Hypersphere with Color Cycling")
    points_4d_hypersphere = efx.generate_4d_hypersphere(num_points=300)
    hypersphere_animation_config = {
        'shape_type': 'hypersphere',
        'rotation_axes': 'xyz',
        'rotation_angles': [30, 45, 20],
        'rotation_speeds': [0.8, 0.6, 0.4], # Default speeds are now relative to slider
        'scale_factor': 1.2,
        'camera_elevation': 35,
        'camera_azimuth': 45,
        'color_cycle_speed': 15 # Adjust color cycle speed
    }
    print("  - Interactive hypersphere animation with XYZ rotation, color cycling, and rotation speed slider...")
    efx.run_animation(points_4d_hypersphere, hypersphere_animation_config, frames=250, interval=40, save_animation=False, filename='interactive_hypersphere.mp4')
    print("  - Interactive hypersphere animation complete.\n")


    # --- Example 2: Interactive Hypercube Shear + Rotation + Different Colormap ---
    print("Example 2: Interactive Hypercube Shear + Rotation + Plasma Colormap")
    points_4d_hypercube = efx.generate_4d_hypercube(num_points_per_edge=15)
    hypercube_animation_config = {
        'shape_type': 'hypercube',
        'transformation_type_4d': 'shear_w',
        'rotation_axes': 'yz',
        'rotation_angles': [20, 30, 0],
        'rotation_speeds': [0.5, 0.7, 0],
        'scale_factor': 0.8,
        'camera_elevation': 25,
        'camera_azimuth': -30,
        'color_cycle_speed': 20 # Different color cycle speed
    }
    efx.colormap = cm.plasma # Change colormap for hypercube example - try 'viridis', 'plasma', 'magma', 'cividis'
    print("  - Interactive hypercube animation with 4D shear, YZ rotation, plasma colormap, and rotation speed slider...")
    efx.run_animation(points_4d_hypercube, hypercube_animation_config, frames=300, interval=40, save_animation=False, filename='interactive_hypercube_shear.mp4')
    print("  - Interactive hypercube animation complete.\n")


    print("--------------------------------------------")
    print("Experiment with the slider to change rotation speed in real-time!")
    print("Modify animation settings and colormaps in the `if __name__ == '__main__':` block for endless variations.")
