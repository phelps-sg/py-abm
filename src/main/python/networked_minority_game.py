from abm import *

import random
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import functools
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.animation import TimedAnimation    

class BarCustomer(IntelligentAgent):
    """A customer at the El Farol Bar"""
  
    memory_size = 2
    
    def __init__(self, name, epsilon=0.3):
        self.decision = 1
        self.payoff = 0
        super(BarCustomer, self).__init__(name, num_states=(2 ** self.memory_size), num_actions=2, epsilon=epsilon)
        
    def calculate_payoff(self, environment):
        if self.decision == environment.minority_decision:
            return 1
        else:
            #return -1
            return 0

    def interact(self, environment):
        self.previous_decision = self.decision
        self.integrateObservation(self.make_observation(environment))
        self.decision = self.make_decision()
        current_payoff = self.calculate_payoff(environment)
        self.giveReward(current_payoff)
        self.payoff += current_payoff
        
    """A customer at the El Farol Bar"""
    def convert_history_to_binary_str(self, environment):
        bit_strings = ["%d" % i for i in environment.history[-self.memory_size:]]     
        # Concatenate all the binary digits in the list into a single string
        return functools.reduce(lambda x, y: x + y, bit_strings)
         
    def make_observation(self, environment):
        return [ int(self.convert_history_to_binary_str(environment), 2) ]
        
    def make_decision(self):
        """Decide whether or not to attend the bar"""
        # For now we make an entirely random decision
        #return random.randint(0,1)
        return self.choose_action()
            
class ElFarolBar(Environment):
    """An environment representing the El Farol Bar"""
 
    def __init__(self, agents):
        # Initialise the history to 10 random bits
        self.history = [ randint(0,1) for i in range(10) ]
        # Initialise the time-series of attendance
        self.attendance = []
        # Initialise the last minority decision to a random bit
        self.minority_decision = randint(0,1)
        # Populate the bar with agents by calling the superclass constructor
        super(ElFarolBar, self).__init__(agents)
        
    def advance_time(self):
        num_agents = len(self.agents)
        number_picking_zero = len([x for x in self.agents if x.decision == 0])
        number_picking_one = num_agents - number_picking_zero
        # Record the attendance at the bar 
        self.attendance.append(number_picking_one)
        # Now save the outcome ready to inform agents
        if number_picking_zero < number_picking_one:
            self.minority_decision = 0
        else:
            self.minority_decision = 1
        # Record the time-series of outcomes
        self.history.append(self.minority_decision)
        # Call the superclass to update the clock etc.
        super(ElFarolBar, self).advance_time()
       

class Visualisation(TimedAnimation):
    """
    An animated visualisation of the simulation.
    """
    
    colors = {
        0:  'g',
        1:  'r'
    }
    
    def __init__(self, model, max_frames=100, interval=100):
        self.max_frames = max_frames
        self.model = model
        self.lines = dict()
        self.node_pos = nx.spring_layout(self.model.get_graph())
        self.create_figure()
        TimedAnimation.__init__(self, self.fig, interval, blit=True)
        
    def plot(self):
        self.create_figure()
        #self.plot_time_series()
        self.plot_network()
        
    def create_figure(self):
        self.fig, self.ax_network = plt.subplots(1,1)
        #self.fig, [self.ax_network, self.ax_ts] = plt.subplots(2,1)
        #self.ax_ts.set_xlim(0, self.model.max_ticks)
        #self.ax_ts.set_xlim(0, 100)
        #self.ax_ts.set_ylim(0, 100)    
        
    def plot_time_series(self):
        for s in ALL_STATES:
            self.lines[s], = self.ax_ts.plot(self.model.time_series[s],
                                                            color=s.color,
                                                            label=s.label)
    def plot_network(self):
        return nx.draw(self.model.get_graph(), self.node_pos, ax=self.ax_network)
    
    def redraw_agents(self): 
         colors = [self.colors[agent.decision] for agent in self.model.population]
         graph = self.model.get_graph()
         return [nx.draw_networkx_nodes(graph, self.node_pos,
     
                                                    nodelist=graph.nodes(),
                                                    node_color=colors,
                                                    ax=self.ax_network)]
    def copy_time_series(self):
        result = dict()
        for s in ALL_STATES:
            result[s] = copy.copy(self.model.time_series[s])
        return result
    
    def redraw_ts_lines(self):
        # Get a safe copy to prevent threading inconsistencies
        time_series = self.copy_time_series()
        for s in ALL_STATES:
            data = time_series[s]
            if len(data) > 1:
                self.lines[s].set_ydata(data)
                self.lines[s].set_xdata(range(0,len(data)))
        return self.lines.values()
    
    def new_frame_seq(self):
        return iter(range(self.max_frames))    
    
    def _init_draw(self):
        #self.plot_time_series()
        self.plot_network()
        
    def _draw_frame(self, tick):
        #self._drawn_artists = self.redraw_ts_lines() + self.redraw_agents()
        self._drawn_artists = self.redraw_agents()
 
num_agents = 101
customers = [BarCustomer("agent#%d" % (i+1), epsilon=0.3) for i in range(num_agents)]
network = nx.random_geometric_graph(num_agents,  0.2)
my_simulation = NetworkedSimulation(Environment(customers), ElFarolBar, 500, network)
visualisatin = Visualisation(my_simulation)
my_simulation.start()
plt.show()
print("done.")
print("Payoffs:")
print([agent.payoff for agent in my_simulation.environment.agents])

