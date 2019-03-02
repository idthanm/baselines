from baselines.ppo1.Environment import vehicle
from baselines.ppo1.Environment import geometry
import random
from matplotlib import pyplot as plt
import numpy as np
from gym import spaces


class Env(object):
    def __init__(self, N, pattern, height, width, laneWidth = 4, laneNum = 1):
        self.N = N
        self.pattern = pattern
        self.safeDist = 0.1
        self.height = height
        self.width = width
        self.laneNum = laneNum
        self.laneWidth = laneWidth
        self.trafficModel = self.generateModel()
        self.reward_model = RewardModel(step_reward=-1, crash_reward=-50, one_passed_reward=10, all_passed_reward=50)
        self.vehs = []
        self.endNum = 0
        self.observation_space = spaces.Box(np.array([-self.height, 0] * self.N), np.array([self.height, 10] * self.N))
        self.action_space = spaces.Box(low=np.array([-5] * self.N), high=np.array([5] * self.N))


    def manualSet(self, modelList):
        self.vehs = []
        state = []
        for i in range(self.N):
            modelTag = modelList[i]
            veh_temp = vehicle.Vehicle(self.trafficModel[modelTag])
            self.vehs.append(veh_temp)
            while self.collisionCheck()[0]:
                self.vehs.pop()
                modelTag = modelList[i]
                veh_temp = vehicle.Vehicle(self.trafficModel[modelTag])
                self.vehs.append(veh_temp)
        self.vehs.sort(key=takeId)
        for i in range(self.N):
            state += [self.vehs[i].getRelPos(), self.vehs[i].vel]
        return np.array(state)


    def reStart(self):
        self.vehs = []
        state = []
        for _ in range(self.N):
            modelTag = random.randint(0, len(self.trafficModel) - 1)
            veh_temp = vehicle.Vehicle(self.trafficModel[modelTag])
            self.vehs.append(veh_temp)

            while self.collisionCheck()[0]:
                self.vehs.pop()
                modelTag = random.randint(0, len(self.trafficModel) - 1)
                veh_temp = vehicle.Vehicle(self.trafficModel[modelTag])
                self.vehs.append(veh_temp)
        self.vehs.sort(key=takeId)
        for i in range(self.N):
            state += [self.vehs[i].getRelPos(), self.vehs[i].vel]
        return np.array(state)


    def collisionCheck(self):
        flag = False
        coupleSave = []

        for i in range(len(self.vehs) - 1):
            for j in range(i+1, len(self.vehs)):
                if not (self.vehs[i].endFlag or self.vehs[j].endFlag):
                    if geometry.rectCheck(self.vehs[i].safeX, self.vehs[i].safeY, self.vehs[j].safeX, self.vehs[j].safeY):
                    # if Euclid(self.vehs[i], self.vehs[j], 0.1):
                        flag = True
                        coupleSave.append((i, j))

        return flag, coupleSave


    def endCheck(self):
        flag = 0

        for veh in self.vehs:
            #if veh.endFlag:
            if veh.getRelPos() < -5:
                flag += 1

        return bool(flag), flag


    def updateEnv(self, action, rewardModel = "single"):
        state = []
        reward = 0
        for i in range(self.N):
            self.vehs[i].stateupdate(action[i])
            state += [self.vehs[i].getRelPos(), self.vehs[i].vel]
            #print(self.vehs[i].trafficModel.id, "|", self.vehs[i].posx, "|", self.vehs[i].ref[3][0], "|",
            #     self.vehs[i].posy, "|", self.vehs[i].ref[3][1],"|", self.vehs[i].endFlag, "|", self.vehs[i].vel)

        collisionFlag = self.collisionCheck()[0]
        endNum = self.endCheck()[1]

        reward += self.reward_model.step_reward
        if collisionFlag:
            reward += self.reward_model.crash_reward
        if rewardModel == "all":
            if endNum == self.N:
                reward += self.reward_model.one_passed_reward
        elif rewardModel == "single":
            if self.endNum != endNum:
                reward += self.reward_model.one_passed_reward
            self.endNum = endNum
            if endNum == self.N:
                reward += self.reward_model.all_passed_reward

        return np.array(state), reward, collisionFlag or endNum == self.N, endNum == self.N


    def showEnv_init(self):
        plt.figure(figsize = (10, 10))
        plt.ion()


    def showEnv(self):
        plt.cla()
        plt.title("Demo")
        ax = plt.axes(xlim = (-self.width - 3, self.width + 3), ylim = (-self.height - 3, self.height + 3))
        plt.axis("equal")
        argument = 1.5
        plt.plot([self.laneWidth, self.laneWidth, self.laneWidth + argument, self.width],
                 [- self.height, - self.laneWidth - argument, - self.laneWidth, - self.laneWidth], "k")
        plt.plot([self.laneWidth, self.laneWidth, self.laneWidth + argument, self.width],
                 [self.height, self.laneWidth + argument, self.laneWidth, self.laneWidth], "k")
        plt.plot([- self.laneWidth, - self.laneWidth, - self.laneWidth - argument, - self.width],
                 [- self.height, - self.laneWidth - argument, - self.laneWidth, - self.laneWidth], "k")
        plt.plot([- self.laneWidth, - self.laneWidth, - self.laneWidth - argument, - self.width],
                 [self.height, self.laneWidth + argument, self.laneWidth, self.laneWidth], "k")

        plt.plot([0.0, 0.0], [- self.height, self.height], ":k")
        plt.plot([- self.width, self.width], [0.0, 0.0], ":k")

        lines = []

        for i in range(self.N):
            if not self.vehs[i].endFlag:
                line = ax.plot([], [], "k")[0]
                lines.append(line)
                lines[-1].set_data(self.vehs[i].boundX, self.vehs[i].boundY)

        safeLine = []
        for i in range(self.N):
            if not self.vehs[i].endFlag:
                line = ax.plot([], [], ":r")[0]
                safeLine.append(line)
                safeLine[-1].set_data(self.vehs[i].safeX, self.vehs[i].safeY)

        for i in range(self.N):
            if not self.vehs[i].endFlag:
                plt.plot(self.vehs[i].middlePoint[0], self.vehs[i].middlePoint[1], "go")

        plt.pause(0.1)


    def generateModel(self):
        result = []
        if self.laneNum == 1:
            RD = (0.5 * self.laneWidth, -self.height); LD = (-0.5 * self.laneWidth, -self.height)
            DR = (self.width, -0.5 * self.laneWidth); UR = (self.width, 0.5 * self.laneWidth)
            RU = (0.5 *self.laneWidth, self.height); LU = (-0.5 * self.laneWidth, self.height)
            UL = (-self.width, 0.5 * self.laneWidth); DL = (-self.width, -0.5 * self.laneWidth)

            i = 0
            result.append(trafficModel(RD, DR, 'DR', i)); i += 1
            result.append(trafficModel(RD, RU, 'DU', i)); i += 1
            result.append(trafficModel(RD, UL, 'DL', i)); i += 1

            result.append(trafficModel(UR, RU, 'RU', i)); i += 1
            result.append(trafficModel(UR, UL, 'RL', i)); i += 1
            result.append(trafficModel(UR, LD, 'RD', i)); i += 1

            result.append(trafficModel(DL, LD, 'LD', i)); i += 1
            result.append(trafficModel(DL, DR, 'LR', i)); i += 1
            result.append(trafficModel(DL, RU, 'LU', i)); i += 1

            result.append(trafficModel(LU, UL, 'UL', i)); i += 1
            result.append(trafficModel(LU, LD, 'UD', i)); i += 1
            result.append(trafficModel(LU, DR, 'UR', i)); i += 1

        return result


class trafficModel(object):
    def __init__(self, start, end, flag, id):
        self.start = start
        self.end = end
        self.flag = flag
        self.id = id

class RewardModel(object):
    def __init__(self, step_reward, crash_reward, one_passed_reward, all_passed_reward):
        self.step_reward = step_reward
        self.crash_reward = crash_reward
        self.one_passed_reward = one_passed_reward
        self.all_passed_reward = all_passed_reward

# def Euclid(veh1, veh2, safeDist):
#     if (veh1.cen_x - veh2.cen_x) ** 2 + (veh1.cen_y - veh2.cen_y) ** 2 < (veh1.R + veh2.R + safeDist) ** 2:
#         return True

def takeId(elem):
    return elem.trafficModel.id