""" A simple class library for agent-based simulation modelling in Python

        (C) 2016 Steve Phelps
        
        http://sphelps.net/
"""     

import pandas

import numpy as np

from random import shuffle

from threading import Thread

from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.learners import Q

class Agent(object):
    """Abstract super-class for all types of Agent."""

    def __init__(self, name):
        self.name = name
    
    def interact(self, environment):
        """
        Sub-classes should override this method and supply
        their own functionality here.
        """
        # The stub implementation simply prints name
        print "%s interacting with %s" % (self.name, environment)


class IntelligentAgent(Agent, LearningAgent):
    """An agent that learns through a value-based RL algorithm"""
  
    def __init__(self, name, num_states, num_actions, epsilon = 0.3, gamma = 0.99, alpha = 0.95):
        self.controller = ActionValueTable(num_states, num_actions)
        self.controller.initialize(np.random.rand(num_states * num_actions))
        self.learner = Q(gamma = gamma, alpha = alpha)
        self.learner.batchMode = False
        self.learner.explorer.epsilon = epsilon
        LearningAgent.__init__(self, self.controller, self.learner)
        Agent.__init__(self, name)
   
    def choose_action(self):
        return self.getAction()[0]
 
 
class Environment(object):
    """A simple environment consisting of agents, and a time value."""

    def __init__(self, agents):
        """Construct an environment populated by the given list of agents"""
        self.agents = agents
        self.time = 0
    
    def advance_time(self):
        self.time += 1
      
class Simulation(Thread):
    """A single realisation of an agent-based model"""

    def __init__(self, environment, max_duration = 100): 
        self.environment = environment
        self.population = environment.agents
        self.is_finished = False
        self.max_duration = max_duration
        super(Simulation, self).__init__()
        
    def single_step(self):
        """Invoke agent interactions"""
        shuffle(self.population)
        for agent in self.population:
            agent.interact(self.environment)
        self.environment.advance_time()
        
    def finished(self):
        """Return true iff the stopping criterion is reached"""
        return self.environment.time >= self.max_duration
        
    def run(self):
        """
        Simulate the model by repeatedly invoking agent interactions until
        the stopping criterion is reached.
        """
        while not self.finished():
            self.single_step()
     
     
class SimulationController(object):
    """
    Manage one or more realisations of a simulation model.
    The controller is responsible for intialising each realisation by
    looking up functions to intiailise each property of the simulation,
    and collects data on each simulation when it finishes.
    The resulting dataset can be retreived as a pandas DataFrame.
    
    argument:  a function returning an instance of Simulation
    
    Keyword arguments:
    n                -- The number of realisations
    data_collectors  -- A dict mapping variable names onto functions of the environment (default {}).
    params           -- A dict mapping parameter names onto no-arg functions for intialising them (default {}).
    """
    
    def __init__(self, sim_factory, n = 100, data_collectors = {}, params = {}):
        self.sim_factory = sim_factory
        self.params = params
        self.n = n
        self.data_collectors = data_collectors
        # Initialise the dataset dictionary
        self.dataset = {}
        for stat in data_collectors:
            self.dataset[stat] = []
        for param in params:
            self.dataset[param] = []
        
    def run(self):
        """
        Run all realisations of the underlying simulation model and collect
        the results.
        """
        for i in range(self.n):
            print "Running simulation %d of %d ..." % ((i+1), self.n)
            self.initialise_parameters()
            self.record_parameters()
            print self.current_params
            sim = self.sim_factory(**self.current_params)
            sim.run()
            print "done."
            self.collect_data(sim)
            
    def collect_data(self, simulation):
        """
        Collect data at the end of the simulation running appending it
        onto the dataset.
        """
        for stat in self.data_collectors:
            fn = self.data_collectors[stat]
            self.dataset[stat].append(fn(simulation.environment))
            
    def record_parameters(self):
        """
        Record the parameters used to intialise each realisation in
        our dataset.
        """
        for param in self.params:
            self.dataset[param].append(self.current_params[param])
            
    def initialise_parameters(self):
        self.current_params = dict(self.params)
        for param_name in self.params:
            initialise_param = self.current_params[param_name]
            self.current_params[param_name] = initialise_param()
             
    def data_frame(self):
        """
        Return the dataset as a pandas DataFrame.  
        The dataset contains parameter values and simulation outcomes.
        """
        return pandas.DataFrame(self.dataset)
    
    
class NetworkedSimulation(Simulation):
    """
        A networked simulation model in which agent interactions occur over a 
        network represented by a NetworkX Graph object.
    """
    
    def __init__(self, environment, env_factory, max_duration, graph):
        super(NetworkedSimulation, self).__init__(environment, max_duration)
        self.graph = graph
        for node_id in self.graph.nodes():
            # Each node on the network contains an agent AND an environment
            attrs = {
                'agent':  self.get_agent(node_id),
                'env':    env_factory([])
            }
            self.graph.add_nodes_from([node_id], **attrs)
            
    def get_agent(self, node_id):
        return self.population[node_id]
     
    def get_graph(self):
        return self.graph
     
    def make_local_interactions(self, node_id):
        agent = self.graph.node[node_id]['agent']
        local_env = self.graph.node[node_id]['env']
        neighbouring_nodes = self.graph.neighbors(node_id)
        neighbors = [self.get_agent(n) for n in neighbouring_nodes]
        local_env.agents = neighbors + [agent]
        for agent in local_env.agents:
            agent.interact(local_env)
        local_env.advance_time()        
            
    def single_step(self):
        for node_id in self.graph.nodes():
            self.make_local_interactions(node_id)
        self.environment.advance_time()
