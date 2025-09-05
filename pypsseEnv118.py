import gym
import random
import pandas as pd
import os,sys
from gym import spaces
from gym.utils import seeding
import numpy as np
import logging
import math
import networkx as nx
import matplotlib.pyplot as plt
import copy
# Path stuff PSSE
sys.path.append(r"C:\Program Files\PTI\PSSE35\35.3\PSSPY39")
sys.path.append(r"C:\Program Files\PTI\PSSE35\35.3\PSSBIN")
sys.path.append(r"C:\Program Files\PTI\PSSE35\35.3\PSSLIB")
sys.path.append(r"C:\Program Files\PTI\PSSE35\35.3\EXAMPLE")

# Importing psspy
import psse35
psse35.set_minor(3)
import psspy
import redirect

from psspy import _i, _f, _s, _o

redirect.psse2py()
psspy.psseinit(10000)


# #################### Environment #######################
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.WARNING)


###################-[25, 26]-[19, 20]
line_list = [[17, 31], [15, 33], [19, 34], [69, 47], [69, 49], [69, 77], [75, 77], [76, 77], [80, 96], [80, 98], [80, 99],
             [82, 96], [89, 92], [91, 92], [96, 97], [23, 24], [30, 38],[5,8], [17, 30], [68, 69], [80, 81]]

line_list0 = [[17, 31], [15, 33], [19, 34], [69, 47], [69, 49], [69, 77], [75, 77], [76, 77], [80, 96], [80, 98], [80, 99],
              [82, 96], [89, 92], [91, 92], [96, 97], [23, 24], [30, 38]]

#######-[59, 63]-[61, 64]-[25, 26]
Trans = [[5,8], [17, 30],[68, 69], [80, 81]]

edges= [[1, 2], [1, 3], [2, 12], [3, 5], [3, 12], [4, 5], [4, 11], [5, 6], [5, 11], [6, 7], 
         [7, 12], [8, 9], [8, 30], [9, 10], [11, 12], [11, 13], [12, 14], [12, 16], [12, 117], 
         [13, 15], [14, 15], [15, 17], [15, 19], [15, 33], [16, 17], [17, 18], [17, 31], [17, 113], 
         [18, 19], [19, 20], [19, 34], [20, 21], [21, 22], [22, 23], [23, 24], [23, 25], [23, 32], 
         [24, 70], [24, 72], [25, 27], [26, 30], [27, 28], [27, 32], [27, 115], [28, 29], [29, 31], 
         [30, 38], [31, 32], [32, 113], [32, 114], [33, 37], [34, 36], [34, 37], [34, 43], [35, 36], 
         [35, 37], [37, 39], [37, 40], [38, 65], [39, 40], [40, 41], [40, 42], [41, 42], [42, 49], 
         [43, 44], [44, 45], [45, 46], [45, 49], [46, 47], [46, 48], [47, 49], [47, 69], [48, 49], 
         [49, 50], [49, 51], [49, 54], [49, 66], [49, 69], [50, 57], [51, 52], [51, 58], [52, 53], 
         [53, 54], [54, 55], [54, 56], [54, 59], [55, 56], [55, 59], [56, 57], [56, 58], [56, 59], 
         [59, 60], [59, 61], [60, 61], [60, 62], [61, 62], [62, 66], [62, 67], [63, 64], [64, 65], 
         [65, 68], [66, 67], [68, 81], [68, 116], [69, 70], [69, 75], [69, 77], [70, 71], [70, 74], 
         [70, 75], [71, 72], [71, 73], [74, 75], [75, 77], [75, 118], [76, 77], [76, 118], [77, 78], 
         [77, 80], [77, 82], [78, 79], [79, 80], [80, 96], [80, 97], [80, 98], [80, 99], [82, 83], 
         [82, 96], [83, 84], [83, 85], [84, 85], [85, 86], [85, 88], [85, 89], [86, 87], [88, 89],
         [89, 90], [89, 92], [90, 91], [91, 92], [92, 93], [92, 94], [92, 100], [92, 102], [93, 94], 
         [94, 95], [94, 96], [94, 100], [95, 96], [96, 97], [98, 100], [99, 100], [100, 101], [100, 103],
         [100, 104], [100, 106], [101, 102], [103, 104], [103, 105], [103, 110], [104, 105], [105, 106], 
         [105, 107], [105, 108], [106, 107], [108, 109], [109, 110], [110, 111], [110, 112], [114, 115], [5,8], 
         [17, 30], [25, 26], [37, 38],[65, 66], [68, 69], [80, 81],[59, 63],[61, 64]]


#----- Creating original graph of the network by adding all the edges
G_original = nx.Graph()
for e in edges:
    u,v = e
    G_original.add_edge(u,v)
nodelist = list(G_original.nodes()) # This nodelist is very important to maintain same order in adjacency matrix

FolderName=os.path.dirname(os.path.realpath("__file__"))
file = r"" + FolderName + "\IEEE118_v33.sav"""
fileSnp = r"" + FolderName + "\IEEE118.snp"""

class pypsseEnv118(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print("Initializing 118-bus env")

        psspy.case(file)
        psspy.rstr(fileSnp)
        _, buses  = psspy.abusint(-1, string="NUMBER")   #read the number of buses
        buses = np.array(buses[0]) # Corrected the error
        n_actions=len(line_list) 
        self.lineOutIndx = 0 # the line outage for the environment
        self.action_space = spaces.MultiBinary(n_actions)
        self.observation_space = spaces.Dict({
            "BusVoltage":spaces.Box(low=0, high=2, shape=(len(buses),),dtype=np.float64),
            "BusAngle":spaces.Box(low=0, high=80, shape=(len(buses),),dtype=np.float64),
            "Adjacency":spaces.Box(low=0, high=2, shape=(len(buses),len(buses)),dtype=np.float64),
            "TotalMismatch":spaces.Box(low=0, high=10000, shape=(1,)),
            "Convergence":spaces.Box(low=0, high=10000, shape=(1,)),
            "IslandFlag":spaces.Box(low=0, high=10000, shape=(1,)),
            "Islanding":spaces.Box(low=0, high=10000, shape=(1,)),
            "IslandFactor":spaces.Box(low=0, high=10000, shape=(1,)),
            }) #Assign a very high value to mismatch
        print('Env initialized')
        
    
    def step(self, action):   
        
        observation = simulate_ckt(self.lineOutIndx, action)

        # Get the current number of islands
        num_islands = observation["Islanding"]
        
        Voltage =  observation ["BusVoltage"]
        # Apply the penalty to elements in the array
        for i in range(len(Voltage)):
            if Voltage[i] > 1.055 or Voltage[i] < 0.945:
                Volt_penalty = 10  # Add 10 to penalty if the condition is met
            else:
                Volt_penalty = 1


        if num_islands > 4:
            penalty = 10
        else:
            penalty = 1

        # Calculate the reward
        #reward = -0.01 * observation['TotalMismatch'] * observation['IslandFlag'] *observation['IslandFactor']* penalty *observation['Convergence']
        reward = -0.01 * observation['TotalMismatch'] * observation['IslandFlag'] *observation['IslandFactor']* penalty*Volt_penalty


        done = True
        info = {"is_success": done,
                "episode": {
                "r": reward,
                "l": 1
            }
            }
        logging.info('Step success')         
        return observation, reward, done, info        


    ######### Reset Function ##############################    
    def reset(self):
        # In reset function just create 1 line outage (for now)
        logging.info('resetting environment...')        
        psspy.case(file)
        psspy.rstr(fileSnp)

        imin = 0       
        imax = len(line_list0) - 1

        num_outages = 3 
        # num_outages = random.randint(0,3)

        outage_lines = random.sample(line_list0, num_outages)
        Gscenario  = nx.Graph()
        for e in edges:
            (u,v) = e
            Gscenario.add_edge(u,v)
        outindx = []

        for line in outage_lines:
            ru, rv = line
            # Change the status of the outage lines in PSSE
            #psspy.branch_chng_3(ru, rv, r"""1""", [0, ru, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "")
            #psspy.branch_chng_3(ru,rv, r"""1""", [_i, ru, _f, _i, _i, _i], [_i, _i, _i, _i, _i, _i, _i, _i, 1, 1, 1, 1], [_i, _i, _i, _i, _i, _i, _i, _i, _i, _i, _i, _i], "")
            psspy.branch_chng_3(ru,rv,r"""1""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f], "")


            outindx.append(line_list0.index([ru, rv])) #append all the indices of the line outages 
            # Remove the edge corresponding to the outage from the graph
            Gscenario.remove_edge(ru,rv)
           
        self.lineOutIndx = outindx
        adjacency_matrix = nx.to_numpy_array(Gscenario, nodelist)
        
        ierrf = psspy.fdns([0,0,0,1,1,0,99,0])
        if ierrf == 0:
            conv_flag=0.01  
        else:
            conv_flag=100
            
        psspy.bus_chng_4(12,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(10,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(25,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(89,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(100,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(65,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(80,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(69,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(66,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(111,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(26,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)

                
        treeobj = psspy.treedat(999)
        i = treeobj['nislands']
        IslandNumber = treeobj['island_busnum']

        ###### Inertia #############

        ierr, (busnums,bustypes)  = psspy.abusint(-1, 1, string=['NUMBER','TYPE'])           
        plantbuses = [busnum for busnum,bustype in zip(busnums,bustypes) if bustype in [2,3]] 
        ierr, (machbuses,) = psspy.amachint(-1, 1, string=['NUMBER'])
        ierr, (machstatus,)= psspy.amachint(-1, 1, string=['STATUS'])
        ierr, (machids,)   = psspy.amachchar(-1,1, string=['ID'])
        machines = zip(machbuses,machids,machstatus)
        machdyn_data = {}
        synchmacs= []
        for mach in machines:
            ierr, machidx = psspy.macind(mach[0], mach[1])
            machdyn_data[mach] = {'index': machidx}
            # model names
            mdlstrs = ['GEN', 'COMP', 'GOV']
            mdlqtys = ['CON']
            for mdl in mdlstrs:
                ierr, mnam = psspy.mdlnam(mach[0], mach[1], mdl)
                if mnam:
                    machdyn_data[mach][mdl.lower()] = {'name': mnam.strip()}
                    for qty in mdlqtys:
                        ierr, ival = psspy.mdlind(mach[0], mach[1], mdl, qty)
                        machdyn_data[mach][mdl.lower()][qty.lower()] = ival
            if not mach[2]: continue
            genmdl = machdyn_data[mach]['gen']['name']
            genmdl = genmdl.strip()
            conidx = machdyn_data[mach]['gen']['con']
            if   genmdl in ['GENDCO', 'GENROE', 'GENROU', 'GENTPJ1']:
                hindx = conidx + 4
            elif genmdl in ['GENSAE', 'GENSAL']:
                hindx = conidx + 3
            elif genmdl == 'GENTRA':
                hindx = conidx + 1
            else:
                print ('Inertialess Machine model found:%s'%genmdl)
                continue
            ierr, rval = psspy.dsrval('CON', hindx)
            machdyn_data[mach]['gen']['inertia'] = rval
            synchmacs.append(mach)
            sysinertia = 0.0
            scale= 0.001    # to get GW-sec.  If scale=1, Sys Inertia is in MW-sec
            print (" Bus   ID\tMacInertia\tTotalInertia[MW-sec]")
            results = []
            for mac in synchmacs:
                busnum = mac[0]
                macid  = mac[1]
                macstat= mac[2]
            
                if macstat:
                    inertia= machdyn_data[mac]['gen']['inertia']
                    ierr, macbase = psspy.macdat(busnum, macid, 'MBASE')
                    macinertia  = inertia*macbase
                    sysinertia += macinertia
                    results.append((busnum, macinertia*scale, sysinertia*scale))
        df = pd.DataFrame(results, columns=['Bus Number', 'Mac Inertia', 'Sys Inertia'])

        each_island_inertia = []

        for sub_list in IslandNumber:
            sub_total = df[df['Bus Number'].isin(sub_list)]['Mac Inertia'].sum()
            each_island_inertia.append(sub_total)


        # Check if all elements in each_island_inertia are greater than 0
        if all(sub_total > 0 for sub_total in each_island_inertia):
            inertiaZ = 1
        else:
            inertiaZ = 10
        ############### End Inertia ###############

        if i >= 2:
            island_flag=0.01
            mva = 0.001*psspy.sysmsm()
        else:
            island_flag=10
            mva = 100*psspy.sysmsm()
                        
        _, volts = psspy.abusreal(-1, string="PU") 
        volts = np.array(volts[0])
        _, angle = psspy.abusreal(-1, string="ANGLED")
        angle = np.array(angle[0])
        #mva = psspy.sysmsm()
        logging.info("reset complete\n")
        
        return {"BusVoltage":volts, "BusAngle":angle, "TotalMismatch": mva, "Convergence":conv_flag,  "IslandFlag":island_flag,
                "Islanding":i, "IslandFactor":inertiaZ, "Adjacency": adjacency_matrix}


    ######### Test Function ##############################    
    def test_model(self):
      
        psspy.case(file)
        psspy.rstr(fileSnp)

        imin = 0       
        imax = len(line_list0) - 1

        num_outages = 3 
        # num_outages = random.randint(0,3)

        outage_lines = random.sample(line_list0, num_outages)
        Gscenario  = nx.Graph()
        for e in edges:
            (u,v) = e
            Gscenario.add_edge(u,v)
        outindx = []

        for line in outage_lines:
            ru, rv = line
            # Change the status of the outage lines in PSSE
            #psspy.branch_chng_3(ru, rv, r"""1""", [0, ru, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "")
            #psspy.branch_chng_3(ru,rv, r"""1""", [_i, ru, _f, _i, _i, _i], [_i, _i, _i, _i, _i, _i, _i, _i, 1, 1, 1, 1], [_i, _i, _i, _i, _i, _i, _i, _i, _i, _i, _i, _i], "")
            psspy.branch_chng_3(ru,rv,r"""1""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f], "")


            outindx.append(line_list0.index([ru, rv])) #append all the indices of the line outages 
            # Remove the edge corresponding to the outage from the graph
            Gscenario.remove_edge(ru,rv)
           
        self.lineOutIndx = outindx
        adjacency_matrix = nx.to_numpy_array(Gscenario, nodelist)
        
        ierrf = psspy.fdns([0,0,0,1,1,0,99,0])
        if ierrf == 0:
            conv_flag=0.01  
        else:
            conv_flag=100
            
        psspy.bus_chng_4(12,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(10,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(25,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(89,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(100,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(65,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(80,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(69,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(66,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(111,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(26,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)

                
        treeobj = psspy.treedat(999)
        i = treeobj['nislands']
        IslandNumber = treeobj['island_busnum']

        ###### Inertia #############

        ierr, (busnums,bustypes)  = psspy.abusint(-1, 1, string=['NUMBER','TYPE'])           
        plantbuses = [busnum for busnum,bustype in zip(busnums,bustypes) if bustype in [2,3]] 
        ierr, (machbuses,) = psspy.amachint(-1, 1, string=['NUMBER'])
        ierr, (machstatus,)= psspy.amachint(-1, 1, string=['STATUS'])
        ierr, (machids,)   = psspy.amachchar(-1,1, string=['ID'])
        machines = zip(machbuses,machids,machstatus)
        machdyn_data = {}
        synchmacs= []
        for mach in machines:
            ierr, machidx = psspy.macind(mach[0], mach[1])
            machdyn_data[mach] = {'index': machidx}
            # model names
            mdlstrs = ['GEN', 'COMP', 'GOV']
            mdlqtys = ['CON']
            for mdl in mdlstrs:
                ierr, mnam = psspy.mdlnam(mach[0], mach[1], mdl)
                if mnam:
                    machdyn_data[mach][mdl.lower()] = {'name': mnam.strip()}
                    for qty in mdlqtys:
                        ierr, ival = psspy.mdlind(mach[0], mach[1], mdl, qty)
                        machdyn_data[mach][mdl.lower()][qty.lower()] = ival
            if not mach[2]: continue
            genmdl = machdyn_data[mach]['gen']['name']
            genmdl = genmdl.strip()
            conidx = machdyn_data[mach]['gen']['con']
            if   genmdl in ['GENDCO', 'GENROE', 'GENROU', 'GENTPJ1']:
                hindx = conidx + 4
            elif genmdl in ['GENSAE', 'GENSAL']:
                hindx = conidx + 3
            elif genmdl == 'GENTRA':
                hindx = conidx + 1
            else:
                print ('Inertialess Machine model found:%s'%genmdl)
                continue
            ierr, rval = psspy.dsrval('CON', hindx)
            machdyn_data[mach]['gen']['inertia'] = rval
            synchmacs.append(mach)
            sysinertia = 0.0
            scale= 0.001    # to get GW-sec.  If scale=1, Sys Inertia is in MW-sec
            print (" Bus   ID\tMacInertia\tTotalInertia[MW-sec]")
            results = []
            for mac in synchmacs:
                busnum = mac[0]
                macid  = mac[1]
                macstat= mac[2]
            
                if macstat:
                    inertia= machdyn_data[mac]['gen']['inertia']
                    ierr, macbase = psspy.macdat(busnum, macid, 'MBASE')
                    macinertia  = inertia*macbase
                    sysinertia += macinertia
                    results.append((busnum, macinertia*scale, sysinertia*scale))
        df = pd.DataFrame(results, columns=['Bus Number', 'Mac Inertia', 'Sys Inertia'])

        each_island_inertia = []

        for sub_list in IslandNumber:
            sub_total = df[df['Bus Number'].isin(sub_list)]['Mac Inertia'].sum()
            each_island_inertia.append(sub_total)


        # Check if all elements in each_island_inertia are greater than 0
        if all(sub_total > 0 for sub_total in each_island_inertia):
            inertiaZ = 1
        else:
            inertiaZ = 10
        ############### End Inertia ###############

        if i >= 2:
            island_flag=0.01
            mva = 0.001*psspy.sysmsm()
        else:
            island_flag=10
            mva = 100*psspy.sysmsm()
                        
        _, volts = psspy.abusreal(-1, string="PU") 
        volts = np.array(volts[0])
        _, angle = psspy.abusreal(-1, string="ANGLED")
        angle = np.array(angle[0])

        
        return {"BusVoltage":volts, "BusAngle":angle, "TotalMismatch": mva, "Convergence":conv_flag,  "IslandFlag":island_flag,
                "Islanding":i, "IslandFactor":inertiaZ, "Adjacency": adjacency_matrix}



    def render(self, mode='human', close=False):
        pass
 
    
##########################################################################

def simulate_ckt(outindx, action): 


    #outindx =[2, 7 , 5]
    #action = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,  1, 0, 0, 1, 0, 1,1, 0, 0, 1]
    
    psspy.case(file)
    psspy.rstr(fileSnp)
    
    Gsc  = nx.Graph()
    for e in edges:
        (u,v) = e
        Gsc.add_edge(u,v)      
            
    line_openidx = [i for i, val in enumerate(action) if val == 1] #list of indices of open lines
    for lidx in line_openidx:
        lu, lv = line_list[lidx]
        Gsc.remove_edge(lu,lv)
    for oidx in outindx:
        ou, ov = line_list0[oidx]
        if Gsc.has_edge(ou,ov):
            Gsc.remove_edge(ou,ov)

    adjacency_matrix =  nx.to_numpy_array(Gsc, nodelist)


    for idx in outindx:
        Uout = line_list[idx]
        #psspy.branch_chng_3(Uout[0], Uout[1], r"""1""", [0, Uout[0], 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "")
        psspy.branch_chng_3(Uout[0],Uout[1],r"""1""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f], "")

    # Now implement the actions
    index_List = [i for i,val in enumerate(action) if val==1]

    for m in index_List:
        if line_list[m] in line_list0:
            Bout = line_list[m]
            psspy.branch_chng_3(Bout[0],Bout[1],r"""1""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f],"")

        if line_list[m] in Trans:
            Bout2 = line_list[m]
            psspy.two_winding_chng_6(Bout2[0],Bout2[1],r"""1""",[0,_i,_i,_i,_i,_i,_i,_i,_i,_i,_i,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f],_s,_s)
    
##    index_List = [i for i,val in enumerate(action) if val==1]
##    for m in index_List:
##        Bout = line_list0[m]      
##        psspy.branch_chng_3(Bout[0],Bout[1],r"""1""",[0,Bout[0],1,0,0,0],[0,0,0,0,0,0,0,0,1,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0],"")


        psspy.bus_chng_4(12,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(10,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(25,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(89,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(100,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(65,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(80,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(69,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(66,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(111,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
        psspy.bus_chng_4(26,0,[2,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)


    treeobj = psspy.treedat(999)
    i = treeobj['nislands']
    IslandNumber = treeobj['island_busnum']


###### Inertia #############

    ierr, (busnums,bustypes)  = psspy.abusint(-1, 1, string=['NUMBER','TYPE'])           
    plantbuses = [busnum for busnum,bustype in zip(busnums,bustypes) if bustype in [2,3]] 
    ierr, (machbuses,) = psspy.amachint(-1, 1, string=['NUMBER'])
    ierr, (machstatus,)= psspy.amachint(-1, 1, string=['STATUS'])
    ierr, (machids,)   = psspy.amachchar(-1,1, string=['ID'])
    machines = zip(machbuses,machids,machstatus)
    machdyn_data = {}
    synchmacs= []
    for mach in machines:
        ierr, machidx = psspy.macind(mach[0], mach[1])
        machdyn_data[mach] = {'index': machidx}
        # model names
        mdlstrs = ['GEN', 'COMP', 'GOV']
        mdlqtys = ['CON']
        for mdl in mdlstrs:
            ierr, mnam = psspy.mdlnam(mach[0], mach[1], mdl)
            if mnam:
                machdyn_data[mach][mdl.lower()] = {'name': mnam.strip()}
                for qty in mdlqtys:
                    ierr, ival = psspy.mdlind(mach[0], mach[1], mdl, qty)
                    machdyn_data[mach][mdl.lower()][qty.lower()] = ival
        if not mach[2]: continue
        genmdl = machdyn_data[mach]['gen']['name']
        genmdl = genmdl.strip()
        conidx = machdyn_data[mach]['gen']['con']
        if   genmdl in ['GENDCO', 'GENROE', 'GENROU', 'GENTPJ1']:
            hindx = conidx + 4
        elif genmdl in ['GENSAE', 'GENSAL']:
            hindx = conidx + 3
        elif genmdl == 'GENTRA':
            hindx = conidx + 1
        else:
            print ('Inertialess Machine model found:%s'%genmdl)
            continue
        ierr, rval = psspy.dsrval('CON', hindx)
        machdyn_data[mach]['gen']['inertia'] = rval
        synchmacs.append(mach)
        sysinertia = 0.0
        scale= 0.001    # to get GW-sec.  If scale=1, Sys Inertia is in MW-sec
        print (" Bus   ID\tMacInertia\tTotalInertia[MW-sec]")
        results = []
        for mac in synchmacs:
            busnum = mac[0]
            macid  = mac[1]
            macstat= mac[2]
            
            if macstat:
                inertia= machdyn_data[mac]['gen']['inertia']
                ierr, macbase = psspy.macdat(busnum, macid, 'MBASE')
                macinertia  = inertia*macbase
                sysinertia += macinertia
                results.append((busnum, macinertia*scale, sysinertia*scale))
    df = pd.DataFrame(results, columns=['Bus Number', 'Mac Inertia', 'Sys Inertia'])

    each_island_inertia = []

    for sub_list in IslandNumber:
        sub_total = df[df['Bus Number'].isin(sub_list)]['Mac Inertia'].sum()
        each_island_inertia.append(sub_total)


    # Check if all elements in each_island_inertia are greater than 0
    if all(sub_total > 0 for sub_total in each_island_inertia):
        inertiaZ = 1
    else:
        inertiaZ = 10

    print('Total Island Inertia =', each_island_inertia)
    print('Name of Islands =', IslandNumber)
    print('Line Outage =', outindx)
    print('Number of islands =', i)

    ############### End Inertia ###############

         
    if i >= 2:     
            island_flag=0.01

            psspy.bus_chng_4(12,0,[3,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            psspy.bus_chng_4(10,0,[3,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            psspy.bus_chng_4(25,0,[3,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            psspy.bus_chng_4(89,0,[3,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            psspy.bus_chng_4(100,0,[3,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            psspy.bus_chng_4(65,0,[3,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            psspy.bus_chng_4(80,0,[3,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            psspy.bus_chng_4(69,0,[3,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            psspy.bus_chng_4(66,0,[3,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            psspy.bus_chng_4(111,0,[3,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
            psspy.bus_chng_4(26,0,[3,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f],_s)
    
            ierrf = psspy.fdns([0,0,0,1,1,0,99,0])

            if ierrf == 0:
                conv_flag=0.01 #converged
                mva = 0.001*psspy.sysmsm()
            else:
                conv_flag=100 #not converged
                mva = 100*psspy.sysmsm()
                  
                
    else:
         island_flag=10 #no islands
         conv_flag=100
         mva = 100*psspy.sysmsm()   ## it should be a large number

     
     # Get observation
    _, volts = psspy.abusreal(-1, string="PU") 
    volts = np.array(volts[0])
    _, angle = psspy.abusreal(-1, string="ANGLED")
    angle = np.array(angle[0])



    return {"BusVoltage":volts, "BusAngle":angle, "TotalMismatch": mva, "Convergence":conv_flag, "IslandFlag":island_flag,
            "Islanding":i, "IslandFactor":inertiaZ, "Adjacency": adjacency_matrix}


