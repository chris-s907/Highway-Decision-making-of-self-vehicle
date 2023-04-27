import math
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Vehicle:
    def __init__(self, position, speed, lane, acceleration=0, sign=0, t=0.1, target_speed=30):
        self.action_viz = 'None'
        self.position = position
        self.speed = speed
        self.acceleration = acceleration
        self.lane = lane
        self.sign = sign
        self.signtime = -1
        self.time_step = 0 
        self.last_signtime = -1
        self.t = t
        self.target_speed = target_speed
        self.CHANGELANE = 0

    def act(self, action):
        self.action_viz = action
        if action == 'vKeeping' or action == 0:
            self.acceleration = 0

        elif action == 'changeLaneR' or action == 1:
            if self.time_step - self.last_signtime > 9:
                self.lane = self.lane - 1
                self.sign = 0
                self.CHANGELANE = 1
            else:
                self.CHANGELANE = 0

        elif action == 'changeLaneL' or action == 2:
            if self.time_step - self.last_signtime > 9:
                self.lane = self.lane + 1
                self.sign = 0
                self.CHANGELANE = 1
            else:
                 self.CHANGELANE = 0

        elif action == 'accelerate' or action == 3:
            self.acceleration = 2

        elif action == 'decelerate' or action == 4:
            self.acceleration = -5

        noise = np.random.normal(0, 0.08)
        self.acceleration += noise

        if self.acceleration > 8:
            self.acceleration = 8

        if self.acceleration < -10:
            self.acceleration = -10

        self.speed += self.acceleration * self.t

        if self.speed > 55:
            self.speed = 55
        if self.speed < 5:
            self.speed = 5

        self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2
        self.time_step += 1

    def DTact(self, action):
        if action == 'maintain':
            action = 0

        elif action == 'changeLaneR':
            self.lane = self.lane - 1
            self.sign = 0

        elif action == 'changeLaneL':
            self.lane = self.lane + 1
            self.sign = 0

        else:
            if action - self.acceleration > 1:
                self.acceleration += 1
            elif action - self.acceleration < -1:
                self.acceleration -= 1
            else:
                 self.acceleration = action

        noise = np.random.normal(0, 0.08)
        self.acceleration += noise

        if self.acceleration > 8:
            self.acceleration = 8
        if self.acceleration < -10:
            self.acceleration = -10

        self.speed += self.acceleration * self.t
        self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2

class RoadManager(object):
    def __init__(self, num_lanes=4):
        self.holding_system = []
        for i in range(num_lanes):
            self.holding_system.append([])

    def add(self, vehicle):
        if vehicle.lane < 0:
            self.holding_system[vehicle.lane+1].append(vehicle)
        elif vehicle.lane > 3:
            self.holding_system[vehicle.lane-1].append(vehicle)
        else:
            self.holding_system[vehicle.lane].append(vehicle)

    def delete(self, vehicle):
        if vehicle.lane < 0:
            self.holding_system[vehicle.lane+1].remove(vehicle)
        elif vehicle.lane > 3:
            self.holding_system[vehicle.lane-1].remove(vehicle)
        else:
            self.holding_system[vehicle.lane].remove(vehicle)

class HighwayEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.car_length = 4.5 # length of the ego car
        self.car_width = 2.0 # width of the ego car
        self.lane_width = 4.0 # width of each lane
        self.num_lanes = 4 # number of lanes
        self.num_obstacles = 35
        self.min_speed = 0.0 # minimum speed limit
        self.max_speed = 55.0 # maximum speed limit
        self.max_acceleration = 2.0 # maximum acceleration
        self.max_deceleration = 5.0 # maximum deceleration
        self.max_lane_change = 1 # maximum number of lanes that can be changed at once
        self.time_step = 0
        self.t = 0.1
        self.max_time_step = 1200
        self.obstacle_speeds = [20, 30, 40]

        self.ego = None
        self.obstacles = []
        self.nearest_obstacles_ahead = []
        self.nearest_obstacles_behind = []

        self.fig, self.ax = plt.subplots(figsize=(10, 5))

        # Initialize the state of the environment
        self.reset()

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(5)
        
        low = np.array([0, 0, 0, -np.inf, 0, -1, -np.inf, 0, -1, 0, 0])
        high = np.array([1, 1, 1, np.inf, 1, 1, np.inf, 1, 1, 1, 1])
        self.observation_space = gym.spaces.Box(low=low, 
                                    high=high, 
                                    dtype=np.float32)

        #-----------Metric----------------
        self.collision_time = 0
        self.outOfBoundary_time = 0
        self.lane_change_time = 0
        self.num_scenario = 0
        self.v_list = []
        self.a_list = [0]

    def reset(self):
        self.time_step = 0

        self.manager = RoadManager(self.num_lanes)

        # Initialize the ego
        self.ego = Vehicle(position=0, speed=np.random.randint(30, 50), acceleration=0, lane=np.random.randint(0, self.num_lanes), sign='none', target_speed=40)
        
        self.manager.add(self.ego)
        
        self.obstacles = []

        # Initialize the obstacle
        for i in range(self.num_obstacles):
            FEASIBLE = False
            while not FEASIBLE:
                position = np.random.uniform(-250, 250)
                lane = np.random.randint(0, self.num_lanes)

                # If the generated obstacle does not collide with the ego and other vehicles, considered FEASIBLE
                if (abs(position - self.ego.position) > 15 or lane != self.ego.lane):
                    for o in self.obstacles:
                        if abs(position - o.position) <= 15 and lane == o.lane:
                            FEASIBLE = False
                            break
                    else:
                        FEASIBLE = True
            speed = np.random.randint(35, 40) if np.random.random()<0.5 else np.random.randint(40, 45)
            obstacle = Vehicle(position, speed, lane, acceleration=0, target_speed=speed)
            self.obstacles.append(obstacle)

            self.manager.add(obstacle)
        # Get nearest obstacles
        self.nearest_obstacles = sorted([o for o in self.manager.holding_system[self.ego.lane] if o.position > self.ego.position], key=lambda o: o.position - self.ego.position)
        # self.nearest_obstacles = sorted(self.obstacles, key=lambda o: (self.ego.position-o.position)**2 + 9*(self.ego.lane-o.lane)**2, reverse=False)[:5]
        return self._get_observation()

    def step(self, action):
        # print(f"Excecuted action: {action}")
        # Increment the step count
        self.time_step += 1
        self.manager.delete(self.ego)
        self.ego.act(action)
        self.manager.add(self.ego)

        # Update each obstacle's state
        for obstacle in self.obstacles:
        
        # --------------------- Decision Tree ------------------------
            FINISH = 0
            CHANGELANE = 1
            if obstacle.signtime == self.time_step: # Time to change lane
                ahead_list = [obs for obs in self.manager.holding_system[obstacle.lane] if obs.position > obstacle.position]
                if len(ahead_list) > 0:
                    obs_ahead = min(ahead_list, key=lambda obs: obs.position - obstacle.position)
                    if obs_ahead is not None and obs_ahead.signtime == obstacle.signtime and obs_ahead.signtime > 0:
                        if obstacle.speed < obstacle.target_speed:
                            obstacle.sign = 0
                            obstacle.signtime = -1
                            obstacle.DTact(0.1)
                        elif obstacle.speed < obstacle.target_speed:
                            obstacle.sign = 0
                            obstacle.signtime = -1
                            obstacle.DTact(-0.1)           
                        else:
                            obstacle.sign = 0
                            obstacle.signtime = -1
                            obstacle.DTact('maintain')
                        FINISH = 1

                if not FINISH:
                    for nearbyObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                        if (abs(nearbyObs.position - obstacle.position) < 9) and ((nearbyObs.position > obstacle.position) and nearbyObs.position - obstacle.position <15): # If change lane, collide
                            # Reset, not finish
                            obstacle.sign = 0
                            obstacle.signtime = -1
                            CHANGELANE = 0
                            break
                    if CHANGELANE:
                        if obstacle.sign == -1:
                            self.manager.delete(obstacle)
                            obstacle.DTact('changeLaneR')
                            self.manager.add(obstacle)
                            
                        else:
                            self.manager.delete(obstacle)
                            obstacle.DTact('changeLaneL')
                            self.manager.add(obstacle)
                        obstacle.signtime = -1 # Reset signtime
                        FINISH = 1 # Done, no further operation needed

            # See if is too close to the obstacles ahead
            if not FINISH:
                dangerObs = []
                for o in self.manager.holding_system[obstacle.lane]:
                    if 0 < o.position - obstacle.position < 50: # Too close
                        dangerObs.append(o)

                    if len(dangerObs) > 0:
                        obs_ahead = min(dangerObs, key=lambda obs:obs.position)

                        if obs_ahead.speed - 3 < obstacle.speed:
                            distance = obs_ahead.position - obstacle.position
                            deceleration = -(obstacle.speed - obs_ahead.speed)**2/(2*(distance))
                            if distance < 15:
                                obstacle.DTact(deceleration - 5)
                            else:
                                obstacle.DTact(deceleration - 0.5)
                            if obstacle.sign != 0: # Already turned on light, do not update the signtime
                                FINISH = 1
                                break

                            if obs_ahead.position - obstacle.position < 30:
                                obstacle.signtime = self.time_step + 20 # Turn on the turn signal

                                if obstacle.lane == 0:
                                    obstacle.sign = 1 # Change lane to left
                                    for nearbyObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                                        # If nearby lane has dangerous obstacle, TURN OFF
                                        if (abs(nearbyObs.position - obstacle.position) < 9) and ((nearbyObs.position > obstacle.position) and nearbyObs.position - obstacle.position <15):
                                            obstacle.sign = 0
                                            obstacle.signtime = -1
                                            break

                                elif obstacle.lane == 3:
                                    obstacle.sign = -1 # Change lane to right
                                    for nearbyObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                                        if (abs(nearbyObs.position - obstacle.position) < 9) and ((nearbyObs.position > obstacle.position) and nearbyObs.position - obstacle.position <15):
                                            obstacle.sign = 0
                                            obstacle.signtime = -1
                                            break
                                
                                else:
                                    obstacle.sign = 1 # Change lane to left
                                    LABEL = 1 # OK to change lane
                                    for nearbyObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                                        if (abs(nearbyObs.position - obstacle.position) < 9) and ((nearbyObs.position > obstacle.position) and nearbyObs.position - obstacle.position <15):
                                            obstacle.sign = 0
                                            obstacle.signtime = -1
                                            LABEL = 0
                                            break
                                    
                                    if LABEL == 0: # Cannot change to left, try changing to right
                                        obstacle.sign = -1
                                        for nearbyObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                                            if (abs(nearbyObs.position - obstacle.position) < 9) and ((nearbyObs.position > obstacle.position) and nearbyObs.position - obstacle.position <15):
                                                obstacle.sign = 0
                                                obstacle.signtime = -1
                                                LABEL = 0
                                                break

                            FINISH = 1 # current obstacle update done, continue to the next obstacle
                            break
                    
            if FINISH:
                continue

            # No obstacles ahead
            if obstacle.speed < obstacle.target_speed:
                obstacle.sign = 0
                obstacle.signtime = -1
                obstacle.DTact(0.1)
            elif obstacle.speed > obstacle.target_speed:
                obstacle.sign = 0
                obstacle.signtime = -1
                obstacle.DTact(-0.1)           
            else:
                obstacle.sign = 0
                obstacle.signtime = -1
                obstacle.DTact('maintain')
            #  --------------------- Decision Tree Ends Here------------------------
        self.v_list.append(self.ego.speed)
        self.a_list.append(self.ego.acceleration)

        # self.nearest_obstacles = sorted(self.obstacles, key=lambda o: (self.ego.position-o.position)**2 + 9*(self.ego.lane-o.lane)**2, reverse=False)[:5]
        if self.ego.lane < 0:
            self.nearest_obstacles_ahead = sorted([o for o in self.manager.holding_system[self.ego.lane+1] if o.position > self.ego.position], key=lambda o: o.position - self.ego.position)
            self.nearest_obstacles_behind = sorted([o for o in self.manager.holding_system[self.ego.lane+1] if o.position < self.ego.position], key=lambda o: self.ego.position - o.position)
        elif self.ego.lane > 3:
            self.nearest_obstacles_ahead = sorted([o for o in self.manager.holding_system[self.ego.lane-1] if o.position > self.ego.position], key=lambda o: o.position - self.ego.position)
            self.nearest_obstacles_behind = sorted([o for o in self.manager.holding_system[self.ego.lane-1] if o.position < self.ego.position], key=lambda o: self.ego.position - o.position)
        else:
            self.nearest_obstacles_ahead = sorted([o for o in self.manager.holding_system[self.ego.lane] if o.position > self.ego.position], key=lambda o: o.position - self.ego.position)
            self.nearest_obstacles_behind = sorted([o for o in self.manager.holding_system[self.ego.lane] if o.position < self.ego.position], key=lambda o: self.ego.position - o.position)
        done = False

        # Check for collisions between the ego and the boundary
        
        if self.ego.lane < 0 or self.ego.lane > 3:
            # print(f"Ego's lane: {self.ego.lane}")
            # print(f"Boundary Collision at timestep {self.time_step}")
            self.outOfBoundary_time += 1
            reward = -60
            done = True 

        # Check for collisions between the ego car and obstacles
        if not done:
            for obstacle in self.obstacles:
                if obstacle.lane == self.ego.lane and abs(obstacle.position - self.ego.position) < self.car_length:
                    # print(f"Obs Collision at timestep {self.time_step}")
                    self.collision_time += 1
                    reward = -100
                    done = True
                    break

        # Reward the ego car for maintaining speed and changing lanes
        if not done:
            reward = self.ego.speed / self.ego.target_speed if self.ego.speed < self.ego.target_speed else (2 - self.ego.speed / self.ego.target_speed)
            reward -= 8/9
            if abs(self.ego.speed - self.ego.target_speed) <= 0.4:
                reward += 1

            if action == 1 or action == 2:
                if self.time_step - self.ego.last_signtime < 15:
                    if self.ego.last_signtime != -1:
                        reward += 5/(self.ego.last_signtime - self.time_step)
                reward += 0.15

            if action == 0:
                reward += 0.032

            if self.ego.speed == self.max_speed and action == 'accelerate_0.8':
                reward -= 1

        if self.time_step > 150:
            done = True

        # Return the observation, reward, done flag, and additional info
        observation = self._get_observation()

        if self.ego.CHANGELANE:
            self.ego.last_signtime = self.time_step

        return observation, reward, done, {}
    
    def _get_observation(self):
        # Get the state of the ego car and obstacles
        observation = [self.ego.speed/self.max_speed, (self.ego.acceleration+5)/7, (self.ego.lane+1)/4]
        if len(self.nearest_obstacles_ahead) == 0 and len(self.nearest_obstacles_behind) == 0:
            observation.extend([200, 0.5, 0, -200, 0.5, 0])
        elif len(self.nearest_obstacles_ahead) == 0:
            observation.extend([200, 0.5, 0, self.nearest_obstacles_behind[0].position-self.ego.position, self.nearest_obstacles_behind[0].speed/self.max_speed, self.nearest_obstacles_behind[0].sign])
        elif len(self.nearest_obstacles_behind) == 0:
            observation.extend([self.nearest_obstacles_ahead[0].position-self.ego.position, self.nearest_obstacles_ahead[0].speed/self.max_speed, self.nearest_obstacles_ahead[0].sign, -200, 0.5, 0])
        else:
            observation.extend([self.nearest_obstacles_ahead[0].position-self.ego.position, self.nearest_obstacles_ahead[0].speed/self.max_speed, self.nearest_obstacles_ahead[0].sign, self.nearest_obstacles_behind[0].position-self.ego.position, self.nearest_obstacles_behind[0].speed/self.max_speed, self.nearest_obstacles_behind[0].sign])
        
        for lane in (self.ego.lane-1, self.ego.lane+1):
            if lane < 0 or lane > 3:
                 observation.extend([0])
            else:
                NOTFEASIBLE = 0
                for nearbyObs in self.manager.holding_system[lane]:
                    if abs(nearbyObs.position - self.ego.position) < 10:
                        NOTFEASIBLE = 1
                observation.extend([NOTFEASIBLE]) 
        
        observation = np.array(observation, dtype=np.float32)
        # Add Gaussian noise to the observation
        noise = np.random.normal(0, 0.08, observation.shape)  # Mean=0, Std=0.08
        noisy_observation = observation + noise

        # Clip the noisy observation to the lower and upper bounds
        lower_bound = np.array([0, 0, 0, -np.inf, 0, -1, -np.inf, 0, -1, 0, 0])
        upper_bound = np.array([1, 1, 1, np.inf, 1, 1, np.inf, 1, 1, 1, 1])
        clipped_observation = np.clip(noisy_observation, lower_bound, upper_bound)

        return clipped_observation
 
    def render(self, mode='human'):
        if mode == 'human':
            # fig, ax = plt.subplots(figsize=(10, 5))
            plt.cla()
            # stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            self.ax.set_xlim([self.ego.position-20, self.ego.position+100])
            self.ax.set_ylim([0, (self.num_lanes * self.lane_width)])
            self.ax.set_xlabel('Position')
            self.ax.set_ylabel('Lane')
            self.ax.set_facecolor('#d3d3d3')  # Set the background color to grey
            self.ax.set_aspect('equal')
            for i in range(self.num_lanes):  # draw lane line
                y = i * self.lane_width
                self.ax.axhline(y=y, color='w', linestyle='--')

            # Plot ego vehicle
            ego_vehicle = patches.Rectangle((self.ego.position - self.car_length / 2, 2 + self.ego.lane * self.lane_width - self.car_width / 2),
                                             self.car_length, self.car_width, fc='b', label='Ego Vehicle')
            self.ax.add_patch(ego_vehicle)

            # Plot obstacles
            for obstacle in self.obstacles:
                obstacle_vehicle = patches.Rectangle((obstacle.position - self.car_length / 2, 2 + obstacle.lane * self.lane_width - self.car_width / 2),
                                                     self.car_length, self.car_width, fc='r', label='Obstacle')
                self.ax.add_patch(obstacle_vehicle)
                if obstacle.sign == 1: # Change lane to left
                    arrow = patches.Arrow(obstacle.position - self.car_length / 4,  2 + obstacle.lane * self.lane_width, 0, 2, width=1, color='yellow')
                    self.ax.add_patch(arrow)
                if obstacle.sign == -1: # Change lane to right
                    arrow = patches.Arrow(obstacle.position - self.car_length / 4,  2 + obstacle.lane * self.lane_width, 0, -2, width=1, color='yellow')
                    self.ax.add_patch(arrow)

            # Plot nearest obstacles
            for obstacle in self.nearest_obstacles_ahead[:]:
                if len(self.nearest_obstacles_ahead) > 0:
                    obstacle_vehicle = patches.Rectangle((obstacle.position - self.car_length / 2, 2 + obstacle.lane * self.lane_width - self.car_width / 2),
                                                            self.car_length, self.car_width, fc='g', label='Nearest Obstacle')
                    self.ax.add_patch(obstacle_vehicle)
                    if obstacle.sign == 1: # Change lane to left
                        arrow = patches.Arrow(obstacle.position - self.car_length / 4,  2 + obstacle.lane * self.lane_width, 0, 2, width=1, color='yellow')
                        self.ax.add_patch(arrow)
                    if obstacle.sign == -1: # Change lane to right
                        arrow = patches.Arrow(obstacle.position - self.car_length / 4,  2 + obstacle.lane * self.lane_width, 0, -2, width=1, color='yellow')
                        self.ax.add_patch(arrow)
                    break

            for obstacle in self.nearest_obstacles_behind[:]:
                if len(self.nearest_obstacles_behind) > 0:
                    obstacle_vehicle = patches.Rectangle((obstacle.position - self.car_length / 2, 2 + obstacle.lane * self.lane_width - self.car_width / 2),
                                                        self.car_length, self.car_width, fc='g', label='Nearest Obstacle')
                    self.ax.add_patch(obstacle_vehicle)
                    if obstacle.sign == 1: # Change lane to left
                        arrow = patches.Arrow(obstacle.position - self.car_length / 4,  2 + obstacle.lane * self.lane_width, 0, 2, width=1, color='yellow')
                        self.ax.add_patch(arrow)
                    if obstacle.sign == -1: # Change lane to right
                        arrow = patches.Arrow(obstacle.position - self.car_length / 4,  2 + obstacle.lane * self.lane_width, 0, -2, width=1, color='yellow')
                        self.ax.add_patch(arrow)    
                    break       
            
            action_idx = int(self.ego.action_viz) if self.ego.action_viz is not 'None' else 'None'
            action_dict = {'None': 'None', 0: 'maintain',1: 'changeLaneR',2: 'changeLaneL',3: 'accelerate_0.08',
                   4: 'decelerate_1.0'}

            plt.title(f'Ego action:{action_dict[action_idx]}\nStep: {self.time_step}, Speed: {self.ego.speed:.2f}, Lane: {self.ego.lane}')
            # plt.show(block=False)
            plt.pause(0.01)

if __name__=='__main__':
    env = HighwayEnv()
    
    # env.reset()
    plt.show(block=False)
    env.render()
    # for i in range(len(env.manager.holding_system)):
    #     laneObs = []
    #     for o in env.manager.holding_system[i]:
    #         laneObs.append(o.position)
    #     laneObs = sorted(laneObs)
    #     print(f"At lane {i}, the positions of the obstacles are {laneObs}")
    # Print the initial state of the ego 
    # print(f"Timestep {env.time_step}:")
    # print(f"Ego's position:{env.ego.position}\nEgo's speed: {env.ego.speed}\nEgo's acc: {env.ego.acceleration}\nEgo's lane: {env.ego.lane}")
    # Print the initial state of the obstacles

    # for i in range(len(env.obstacles)):
    #     obs = env.obstacles[i]
    #     print(f"Vehicle_{i}'s position:{obs.position}\nVehicle_{i}'s speed: {obs.speed}\nVehicle_{i}'s lane: {obs.lane}")

    obs, reward, done, _ = env.step(('maintain', 0))
    # print(f"Timestep {env.time_step}:")
    # print(f"Ego's position:{env.ego.position}\nEgo's speed: {env.ego.speed}\nEgo's acc: {env.ego.acceleration}\nEgo's lane: {env.ego.lane}\n")
    # for i in range(len(env.nearest_obstacles)):
    #     obs = env.nearest_obstacles[i]
    #     print(f"Nearest vehicle_{i}'s position:{obs.position}\nVehicle_{i}'s speed: {obs.speed}\nVehicle_{i}'s lane: {obs.lane}\n")

    print(f"Reward = {reward}, done = {done}")
    # for i in range(len(env.manager.holding_system)):
    #     laneObs = []
    #     for o in env.manager.holding_system[i]:
    #         laneObs.append(o.position)
    #     laneObs = sorted(laneObs)
    #     print(f"At lane {i}, the positions of the obstacles are {laneObs}")