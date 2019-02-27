from baselines.ppo1.Environment import environment
import time

height, width = 30, 30
vehNum = 3
vehModelList = [2, 4, 10]


CrossRoad = environment.Env(vehNum, height, width, 4)
CrossRoad.showEnv_init()

for count in range(10000):
    collisionFlag = False
    endFlag = False
    tag = 0
    CrossRoad.manualSet(vehModelList)
    # CrossRoad.reStart()
    print()
    while not endFlag:
        action = [0] * vehNum
        [state, reward, endFlag, _] = CrossRoad.updateEnv(action)
        CrossRoad.showEnv()
        tag += 1
        # print(count, "step: ", tag, "collision?: ", collisionFlag, "end?: ", endFlag)
        # print(reward)
        #print(state)
        #time.sleep(2)


