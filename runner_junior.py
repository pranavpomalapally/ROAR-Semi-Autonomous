from ROAR_Junior_Racing.junior_runner import JuniorRunner
from ROAR_Junior_Racing.configurations.junior_config import JuniorConfig
from ROAR.agent_module.forward_only_agent import ForwardOnlyAgent
from pathlib import Path
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.utilities_module.vehicle_models import Vehicle
import logging
import argparse

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("-auto", action='store_true', help="Enable auto control")
    args = parser.parse_args()

    agent_config_file_path = Path("ROAR/configurations/junior/junior_agent_configuration.json")
    junior_config_file_path = Path("ROAR_Junior_Racing/configurations/configuration.json")

    agent_config = AgentConfig.parse_file(agent_config_file_path)
    junior_config: JuniorConfig = JuniorConfig.parse_file(junior_config_file_path)

    agent = ForwardOnlyAgent(vehicle=Vehicle(), agent_settings=agent_config, should_init_default_cam=True)
    runner = JuniorRunner(agent=agent, config=junior_config)
    runner.start_game_loop(auto_pilot=args.auto)

