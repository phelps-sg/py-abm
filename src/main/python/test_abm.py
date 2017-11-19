import pytest
from unittest import TestCase
from mock import Mock
from abm import Agent, Environment, Simulation

@pytest.mark.unittests
class TestAgent(TestCase):

    @pytest.fixture(autouse=True)
    def setup(self):
        pass

    def test_agent_name(self):
        agent = Agent(name='turtle')
        assert agent.name == 'turtle'

@pytest.mark.unittests
class TestSimulation(TestCase):

    @pytest.fixture(autouse=True)
    def setup(self):
        agent = Mock
        agent.interact = Mock()
        agent_group = [agent(name='1'), agent(name='2'), agent(name='3')]
        self.environment = Environment(agents=agent_group)

    def test_run_simulation(self):
        simulation = Simulation(self.environment, max_duration=100)
        simulation.start()
        while simulation.is_finished == False:
            pass
        assert self.environment.time == 100
