from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random

# Our agent class
class ZergAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ZergAgent, self).__init__()

        self.attack_coordinates = None

    # Core method for our agent, it's where all of our decision making takes
    # place. At the end of every step it returns an action.
    def step(self, obs):
        super(ZergAgent, self).step(obs)

        # Checks if it's the first step of the game.
        if obs.first():
            # We get the centre x and y coordinates of our units on the minimap.
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                    features.PlayerRelative.SELF).nonzero()
            xmean=player_x.mean()
            ymean=player_y.mean()

            if xmean <= 31 and ymean <= 31:
                self.attack_coordinates = (49, 49)
            else:
                self.attack_coordinates = (12, 16)

            zerlings = self.get_units_by_type(obs, units.Zerg.Zergling)
            if len(zerlings) >= 10:
                if self.unit_type_is_selected(obs, units.Zerg.Zergling):
                    if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                        return actions.FUNCTIONS.Attack_minimap("now",
                                                                 self.attack_coordinates)
                
                if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                    return actions.FUNCTIONS.select_army("select")

        spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
        # If there's no spawning pool, it will build one.
        if len(spawning_pools) == 0:
            # Check if we have a Drone selected.
            if self.unit_type_is_selected(obs, units.Zerg.Drone):
                # Check if we can build a Spawning Pool. If we don't have
                # enough minerals this may not be possible and will result
                # in a crash.
                if self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
                    # Selects a random point on the screen.
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)

                    return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x,y))

            # List of all Drones on the screen.
            drones = self.get_units_by_type(obs, units.Zerg.Drone)

            if len(drones) > 0:
                drone=random.choice(drones)
                # The select_all_type parameter here acts like CTRL + click,
                # so all Drones on the screen will be selected.
                return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y))

        # Creates some Zerlings.
        if self.unit_type_is_selected(obs, units.Zerg.Larva):
            #We can spawn an Overlord if we have no free supply
            free_supply = (obs.observation.player.food_cap -
                            obs.observation.player.food_used)

            if free_supply == 0:
                if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
                    return actions.FUNCTIONS.Train_Overlord_quick("now")

            if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                return actions.FUNCTIONS.Train_Zergling_quick("now")

        # Selects all Larva on the screen
        larvae = self.get_units_by_type(obs, units.Zerg.Larva)
        if len(larvae) > 0:
            larva = random.choice(larvae)

            return actions.FUNCTIONS.select_point("select_all_type", (larva.x,
                                                                    larva.y))
            
        return actions.FUNCTIONS.no_op()

    # Utility method that check we have a certain unit_type selected
    # Checks both, the single and multi-selections to see if the first
    # selected unit is the correct type.
    def unit_type_is_selected(self, obs, unit_type):
        if(len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
            return True
        
        if(len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
            return True
        
        return False

    # Utility methid for selecting the units with a given unit type.
    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    # Utility method to check if it can perform an action
    def can_do(self, obs, action):
        return action in obs.observation.available_actions

def main(useless_argv):
    agent = ZergAgent()
    
    try:
        while True:
            with sc2_env.SC2Env(
                map_name="Simple64",
                # Here we specify that the first player is our agent, and the
                # agent's race is Zerg. Then we specify that the second player
                # is a bot, using the game's internal AI, the bot race is
                # random and the difficulty level is very easy.
                players=[sc2_env.Agent(sc2_env.Race.zerg),
                        sc2_env.Bot(sc2_env.Race.random,
                        sc2_env.Difficulty.very_easy)],
                # Here we specify the screen and minimap resolutions. These
                # resolutions essentially determine how many "pixels" of data
                # are in each feature layer, those layers include things like
                # terrain height, visibility, and unit ownership. It also
                # enables feature units.
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True
                ),
                # This parameter determines how many "game steps" pass before
                # your bot will choose an action to take. By defualt is set to 8.
                step_mul=16,
                # Fixed length of each game, set to 0 allows the game to run
                # as long as necessary.
                game_steps_per_episode=0,
                # This parameter is optional,however it is handy for us to see
                # the visualisation as it contains details about all of the
                # observation layers available to your bot.
                visualize=True
            ) as env:
                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)