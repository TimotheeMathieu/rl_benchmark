from benchopt import BaseObjective, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    

# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "rlberry"

    # URL of the main repo for this benchmark.
    url = "https://github.com/rlberry-py/rlberry"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {}

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    requirements = []

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(self, env, rng):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.env = env
        self.rng = rng

    def get_objective(self):
        "Returns a dict to pass to the set_objective method of a solver."
        return dict(env = self.env, rng=self.rng)

    def evaluate_result(self, evaluation):
        """Compute the objective value given the output of a solver.

        The arguments are the keys in the result dictionary returned
        by ``Solver.get_result``.
        """
        return dict(value=evaluation)

    def get_one_result(self):
        "Return one solution for which the objective can be evaluated."
        return dict(evaluations=np.zeros(10))
    

