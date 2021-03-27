import random
from Utils import Utils
from Controllers import Controller
from NonlinearModel import NonlinearModel
import math

u = Utils()
controller = Controller()


class Env:
    def __init__(self):
        self.info = ""
        self.dtau = 0.01
        self.dt = 0.0001
        self.Nsolver = int(self.dtau / self.dt)

        self.target = self.__calculate_random_target()

        self.obs_space = 6
        self.action_space = 3
        self.action_space_max = 30
        self.action_space_min = -30
        self.states = self.__initial_states()
        self.obs_states = self.__obs_calc()

        self.action_to_controller = [0, 0, 0, u.feetTometer(176)]
        self.index = 0
        self.distance = 500
        # self.errsum=[0,0,0,0]
        # self.lasterr=[0,0,0,0]

    def reset(self):
        self.states = self.__initial_states()
        self.obs_states = self.__obs_calc()
        self.action_to_controller = [0, 0, 0, u.feetTometer(176)]
        self.index = 0
        self.target = self.__calculate_random_target()
        # self.errsum=[0,0,0,0]
        # self.lasterr=[0,0,0,0]
        return self.obs_states

    def step(self, action):
        converted_action = self.__action_converter(action)
        next_states = self.__make_action(converted_action)

        self.states = next_states
        self.obs_states = self.__obs_calc()
        done = self.__done_calc()
        reward = self.__reward_calc()

        return self.obs_states, reward, done

    def __initial_states(self):
        return [u.feetTometer(176), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4500.0, 0.0, 0.0, 0.0]

    def __obs_calc(self):
        xd = self.target[0]
        yd = self.target[1]
        zd = self.target[2]

        x = self.states[10]
        y = self.states[11]
        z = self.states[12]

        phi = u.radTodeg(self.states[3])
        theta = u.radTodeg(self.states[4])
        psi = u.radTodeg(self.states[5])

        delta_x = xd - x
        delta_y = yd - y
        delta_z = zd - z

        obs = [delta_x, delta_y, delta_z, phi, theta, psi]

        return obs

    def __action_converter(self, action):
        #action = action[0]
        for i in range(3):
            self.action_to_controller[i] = u.degTorad(action[i])
        return self.action_to_controller

    def __make_action(self, actions):
        new_states = self.states
        for i in range(10):
            U = controller.BacksteppingController(new_states, actions)  # self.errsum,self.lasterr,self.dtau
            # U=data[0]
            # self.errsum=data[1]
            # self.lasterr=data[2]
            new_states = NonlinearModel(new_states, U, self.Nsolver, self.dt)
            self.index += 1
        return new_states

    def __done_calc(self):
        donef = False
        xd = self.target[0]
        yd = self.target[1]
        zd = self.target[2]

        x = self.states[10]
        y = self.states[11]
        z = self.states[12]

        delta_x = xd - x
        delta_y = yd - y
        delta_z = zd - z

        self.distance = math.sqrt(pow(delta_x, 2) + pow(delta_y, 2) + pow(delta_z, 2))

        if self.distance < 10:
            donef = True

        if self.index == 1000:
            donef = True
        return donef

    def __reward_calc(self):
        K=150.31
        n=1.177
        reward =K/(pow(self.distance,n))
        return reward

    def __calculate_random_target(self):
        x = random.uniform(400, 500)
        y = random.uniform(-200, 200)
        z = random.uniform(-200, 200)
        target = [x, y, z]
        return target