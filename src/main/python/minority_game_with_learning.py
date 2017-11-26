import logging
logging.basicConfig(level=logging.DEBUG)
from abm import *
import functools
from numpy.random import randint
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.tools.plotting import scatter_matrix

    
class BarCustomer(IntelligentAgent):
    """A customer at the El Farol Bar"""
  
    memory_size = 5
    
    def __init__(self, name, epsilon=0.3, alpha=0.9):
        self.decision = randint(0, 2)
        self.payoff = 0
        super(BarCustomer, self).__init__(name, num_states=(2 ** self.memory_size), num_actions=2, epsilon=epsilon, alpha=alpha)
        
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
        return self.choose_action()


class ElFarolBar(Environment):
    """An environment representing the El Farol Bar"""
 
    def __init__(self, agents):
        # Initialise the history to 10 random bits
        self.history = [ randint(0, 2) for i in range(10) ]
        # Initialise the time-series of attendance
        self.attendance = []
        # Initialise the last minority decision to a random bit
        self.minority_decision = randint(0, 2)
        # Populate the bar with agents by calling the superclass constructor
        super(ElFarolBar, self).__init__(agents)
        
    def advance_time(self):
        num_agents = len(self.agents)
        number_picking_zero = len([x for x in self.agents if x.decision == 0])
        number_picking_one = num_agents - number_picking_zero
        # Record the attendance at the bar 
        self.attendance.append(number_picking_one - number_picking_zero)
        # Now save the outcome ready to inform agents
        if number_picking_zero < number_picking_one:
            self.minority_decision = 0
        else:
            self.minority_decision = 1
        # Record the time-series of outcomes
        self.history.append(self.minority_decision)
        # Call the superclass to update the clock etc.
        super(ElFarolBar, self).advance_time()


class ElFarolBarSimulation(Simulation):
    
    def __init__(self, num_agents=101, epsilon=0.5,
                     alpha=0.9, memory_size=5):
        BarCustomer.memory_size = memory_size
        customers = \
            [BarCustomer("agent#%d" % (i+1), epsilon=epsilon, alpha=alpha) \
                for i in range(num_agents)]
        environment = ElFarolBar(customers)
        super(ElFarolBarSimulation, self).__init__(environment, 
                                                    max_duration=1000)


logging.info("Running a single simulation...")
my_simulation = ElFarolBarSimulation()
my_simulation.run()
logging.info("done.")

logging.info("Payoffs:")
logging.info([agent.payoff for agent in my_simulation.environment.agents])
logging.info("Attendance time-series:")
plt.plot(my_simulation.environment.attendance)
plt.show()

# Functions to intialise parameters for each realisation
params = { 
    'epsilon':      lambda: np.random.uniform(0, 1),
    'alpha':        lambda: np.random.uniform(0, 1),
    'memory_size':  lambda: np.random.randint(1, 8+1)
}

def autocorr(x, t=1):
    return np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))[0,1]

# Data to collect from each realisation
data_collectors = { 
    'at_mean':  lambda env: np.mean(env.attendance),
    'at_var':   lambda env: np.var(env.attendance),
    'ata':      lambda env: abs(autocorr(env.attendance))
}

# Run 100 simulations with the above parameters and data
sim_controller = SimulationController(ElFarolBarSimulation, 100, data_collectors, params)
sim_controller.run()

# Return the resulting dataset as a data frame
data = sim_controller.data_frame()
logging.info(data)

########################
# Analyse the results
########################

# Produce scatter plots for every pair of variables
scatter_matrix(data)

# Subset of the data containing parameters
independent_variables = data[[k for k in params.keys()]]

# Perform multivariate regression on each dependent variable
for dependent_var_name in data_collectors.keys():
    Y = data[dependent_var_name]
    X = independent_variables
    X = sm.add_constant(X)
    regression_results = sm.OLS(Y, X).fit()
    logging.info(regression_results.summary())
