from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import NoCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from rlberry.agents.torch import DQNAgent
    from rlberry.manager import ExperimentManager, evaluate_agents
    from rlberry import logger
    import pandas as pd

# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'DQN'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.

    stopping_criterion = NoCriterion(strategy="callback")
    
    parameters = {'fit_budget_between_evals':[5000],
                  }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = ['pip:rlberry', 'pip:torch']

    def set_objective(self, env, rng):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        dqn_init_kwargs = dict(gamma=0.99,
                               batch_size=32,
                               chunk_size=8,
                               lambda_=0.5,
                               target_update_parameter=0.005,
                               learning_rate=1e-3,
                               epsilon_init=1.0,
                               epsilon_final=0.1,
                               epsilon_decay_interval=20_000,
                               train_interval=10,
                               gradient_steps=-1,
                               max_replay_size=200_000,
                               learning_starts=5_000,
                               )
        self.manager = ExperimentManager(
            DQNAgent,
            env,
            fit_budget = self.fit_budget_between_evals,
            init_kwargs=dqn_init_kwargs,
            eval_kwargs=dict(eval_horizon=500),
            n_fit=1,
            seed = rng.randint(2**31)
        )        

    def run(self, callback):
        # optimizer and lr schedule init
        max_epochs = callback.stopping_criterion.max_runs
        # Initial evaluation
        while callback():
            self.manager.fit()

    def get_result(self):
        dfs = self.manager.get_writer_data()
        if dfs is None:
            return {'evaluation': 1}
        else:
            df = dfs[0] # there is only one agent
            rewards = df.loc[df['tag']=="episode_rewards", "value"].values
            return {"evaluation": np.mean(rewards[-10:])}
        

