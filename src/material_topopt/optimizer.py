"""A module containing the MMA-based optimizer for the material.

The class MaterialOptimizer is defined, which using the Method of Moving Asymptotes to solve the overall material
optimization problem for minimizing the macroscale compliance when subject to a constraint on the volume fraction
of stiff material.

Typical usage example:

  >>> import matplotlib.pyplot as plt
  >>> material_optimizer = MaterialOptimizer(...)
  >>> material_optimizer.macroscale_problem.plot_mesh()
  >>> material_optimizer.macroscale_problem.representative_volume_element.plot_mesh()
  >>> material_optimizer.run()
  >>> material_optimizer.plot_history()
  >>> plt.show()
"""
import os
import shutil
import logging
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import src.material_topopt.utilities as utils
from src.material_topopt.macroscale_problem import MacroscaleProblem2D
from src.material_topopt.MMA import mmasub


#######################################################################################################################
#######################################################################################################################
class MaterialOptimizer:
    """Material optimizer using the Method of Moving Asymptotes.

    Class that controls the high level execution of the material optimization problem, provided the
    parameters associated with the representative volume element, macroscale boundary value problem,
    and the optimizer. Note that the code currently only is implemented for 2D plane stress problems
    in which the underlying material model is linear elastic and only 3-node triangular finite
    elements are used.

    Attributes:
        macroscale_problem: A MacroscaleProblem2D object.
        logged_values_data_frame: A pandas DataFrame of values logged during the optimization process.
    """

    ###################################################################################################################
    ###################################################################################################################
    def __init__(self,
                 optimization_problem_parameters: dict = None,
                 macroscale_problem_parameters: dict = None,
                 representative_volume_element_parameters: dict = None):
        """Initialization of the optimizer.

        Takes in three dictionaries of parameters. One for the optimizer, macroscale problem, and RVE respectively.

        Args:
            optimization_problem_parameters: A dictionary of optimizer parameters.
            macroscale_problem_parameters: A dictionary of macroscale problem parameters.
            representative_volume_element_parameters: A dictionary of representative volume element parameters.

        Returns:
            None
        """
        self.logger = logging.getLogger("MaterialTopOpt.MaterialOptimizer")
        self.macroscale_problem = \
            MacroscaleProblem2D(macroscale_problem_parameters, representative_volume_element_parameters)
        self.__set_user_specified_parameters_or_defaults(optimization_problem_parameters)
        self.logged_values_data_frame = None


    ###################################################################################################################
    ###################################################################################################################
    def __set_user_specified_parameters_or_defaults(self, optimization_problem_parameters: dict):
        """Set internal attributes using user-specified parameters.

        Checks user-specified parameters and sets internal attributes using either these quantities or defaults.

        Args:
            optimization_problem_parameters: A dictionary of optimizer parameters.

        Returns:
            None
        """
        self.__maximum_number_of_iterations = utils.get_parameter_or_default(
            optimization_problem_parameters, "maximum number of iterations", int, bounds=(1, None),
            default_value=200, required=False)
        self.__volume_fraction_upper_bound = utils.get_parameter_or_default(
            optimization_problem_parameters, "volume fraction constraint upper bound", float, bounds=(0.0, 1.0),
            default_value=0.5, required=False)
        self.__restart_iteration_number = utils.get_parameter_or_default(
            optimization_problem_parameters, "restart iteration number", int, bounds=(0, None),
            default_value=0, required=False)
        self.__restart_file_write_frequency = utils.get_parameter_or_default(
            optimization_problem_parameters, "restart file write frequency", int, bounds=(1, None),
            default_value=10000, required=False)
        self.__mma_move_limit = utils.get_parameter_or_default(
            optimization_problem_parameters, "MMA move limit", float, bounds=(0.01, 1.0),
            default_value=0.15, required=False)
        self.__optimization_history_filepath = utils.get_parameter_or_default(
            optimization_problem_parameters, "optimization history output filepath", str, bounds=(None, None),
            default_value="./MaterialTopOptData.csv", required=False)
        self.__optimization_restart_file_directory = os.path.dirname(
            os.path.abspath(self.__optimization_history_filepath)
            )


    ###################################################################################################################
    ###################################################################################################################
    def __compute_objective_scale_factor(self, current_scale_factor: float, current_unscaled_objective_value: float):
        """Computes a new objective scale factor.

        Computes an objective scale factor to ensure that the scaled objective remains between 1 and 8. This helps
        ensure the scale of the objective is on the order of the constraint.

        Args:
            current_scale_factor: A float representing the current objective scale factor.
            current_unscaled_objective_value: A float representating the current unscaled value.

        Returns:
            A float representing the new objective scale factor.
        """
        assert current_scale_factor > 0.0, "scale factor cannot be zero or negative"
        assert current_unscaled_objective_value > 0.0,\
            "The compliance cannot be zero or negative unless the boundary conditions are incorrectly specified." + \
                f" Returned value = {current_unscaled_objective_value:0.3e}."
        current_scaled_objective_value = current_scale_factor * current_unscaled_objective_value
        if 1.0 < current_scaled_objective_value < 8.0:
            return current_scale_factor
        new_objective_scale_factor = 2.0 / current_unscaled_objective_value
        return new_objective_scale_factor


    ###################################################################################################################
    ###################################################################################################################
    def __get_function_values_and_gradients(self,
                                            design_variables: np.ndarray,
                                            iteration_number: int,
                                            objective_scale_factor: float = 1.0):
        """Run the forward problem and compute the function values and gradients.

        Runs the entire forward analysis and sensitivity analysis for the objective and constraint functions. Then
        current optimization iteration's important quantities are logged and the relevant quantities returned.

        Args:
            design_variables: A numpy array of current design variables.
            iteration_number: An integer that is the current optimization iteration number.
            objective_scale_factor: A float representing the current objective scale factor.

        Returns:
            A tuple (a, b, c, d, e) where 'a' is the unscaled objective value, 'b' is the scaled objective gradient,
            'c' is the scaled constraint value, 'd' is the scaled constraint gradient, and 'e' is the current
            objective scale factor.
        """
        self.macroscale_problem.run_macroscale_forward_analysis(design_variables.ravel(),
                                                                optimization_iteration_number=iteration_number)
        objective_value, objective_gradient = \
            self.macroscale_problem.get_macroscale_compliance_value_and_sensitivity()
        constraint_value, constraint_gradient = self.macroscale_problem.get_volume_fraction_value_and_sensitivity()

        current_objective_scale_factor = self.__compute_objective_scale_factor(objective_scale_factor, objective_value)

        constraint_upper_bound = self.__volume_fraction_upper_bound

        logged_values_dictionary = {"Optimization Iteration Number": iteration_number,
                                    "Macroscale Compliance (objective)": objective_value,
                                    "Objective Scale Factor": current_objective_scale_factor,
                                    "Volume Fraction (constraint)": constraint_value,
                                    "Constraint Upper Bound": constraint_upper_bound}
        updated_logged_values_dictionary = self.macroscale_problem.update_logged_values(logged_values_dictionary)
        if self.logged_values_data_frame is None:
            self.logged_values_data_frame = pd.DataFrame(data=updated_logged_values_dictionary, index=[0])
        else:
            new_data_frame = pd.DataFrame(data=updated_logged_values_dictionary, index=[0])
            self.logged_values_data_frame = \
                pd.concat((self.logged_values_data_frame, new_data_frame), ignore_index=True)


        unscaled_objective_value = float(objective_value)
        scaled_objective_gradient = np.zeros((objective_gradient.size, 1))
        scaled_objective_gradient[:, 0] = current_objective_scale_factor * objective_gradient

        scaled_constraint_value = (float(constraint_value) - constraint_upper_bound) / abs(constraint_upper_bound)
        scaled_constraint_gradient = np.zeros((1, constraint_gradient.size))
        scaled_constraint_gradient[0, :] = (1.0 / abs(constraint_upper_bound)) * constraint_gradient
        return unscaled_objective_value, scaled_objective_gradient, scaled_constraint_value, \
            scaled_constraint_gradient, current_objective_scale_factor


    ###################################################################################################################
    ###################################################################################################################
    def run(self):
        """Runs the optimization problem.

        Args:
            None

        Returns:
            None
        """
        initial_design_variables = self.macroscale_problem.get_initial_design_variables()

        number_of_constraints = 1
        number_of_design_variables = initial_design_variables.size
        eeen = np.ones((number_of_design_variables, 1))
        eeem = np.ones((number_of_constraints, 1))
        zeron = np.zeros((number_of_design_variables, 1))
        zerom = np.zeros((number_of_constraints, 1))
        xval = 0.0 * zeron
        xval[:, 0] = initial_design_variables
        xold1 = xval.copy()
        xold2 = xval.copy()
        xmin = 0.0 * eeen
        xmax = 1.0 * eeen
        low = xmin.copy()
        upp = xmax.copy()
        c = 1000.0*eeem
        d = eeem.copy()
        a0 = 1.0
        a = zerom.copy()
        move = self.__mma_move_limit
        current_objective_scale_factor = 1.0
        iteration_number = 0

        if self.__restart_iteration_number > 0:
            restart_filename = os.path.join(self.__optimization_restart_file_directory,
                                            f"mma_restart_iteration_{self.__restart_iteration_number:04d}.npz")
            if not os.path.exists(restart_filename):
                raise IOError(f"The restart file '{restart_filename}' does not exist!")
            data = np.load(restart_filename)
            iteration_number = self.__restart_iteration_number
            xval = data['xval']
            xold1 = data['xold1']
            xold2 = data['xold2']
            low = data['low']
            upp = data['upp']
            current_objective_scale_factor = float(data['objective_scale_factor'])
            if os.path.exists(self.__optimization_history_filepath):
                try:
                    self.logged_values_data_frame = pd.read_csv(self.__optimization_history_filepath)
                    optimization_iteration_numbers = self.logged_values_data_frame["Optimization Iteration Number"]
                    mask = optimization_iteration_numbers < self.__restart_iteration_number
                    self.logged_values_data_frame = self.logged_values_data_frame[mask]
                except FileNotFoundError as exc:
                    self.logger.warning(exc)
                    message = "Restart requested but unable to read the data file " + \
                        f"'{self.__optimization_history_filepath}' in order to append to the data frame. " + \
                        "Overwriting with new data."
                    self.logger.warning(message)
                    self.logged_values_data_frame = None

        # The iterations start
        while iteration_number < self.__maximum_number_of_iterations:

            # Compute the function values and gradients for current set of design variables
            unscaled_objective_value, scaled_objective_gradient, scaled_constraint_value, \
                scaled_constraint_gradient, current_objective_scale_factor = \
                self.__get_function_values_and_gradients(xval, iteration_number,
                                                                objective_scale_factor=current_objective_scale_factor)
            message = (f"Iteration {iteration_number:4d}, Objective {unscaled_objective_value:11.4e}, Constraint"
              f" {scaled_constraint_value:11.4e} <= 0, Objective Scale Factor {current_objective_scale_factor:11.4e}")
            self.logger.info(message)

            # The MMA subproblem is solved at the current design variable values, xval:
            xmma, _, _, _, _, _, _, _, _, low, upp = \
                mmasub(number_of_constraints, number_of_design_variables, iteration_number, xval, xmin, xmax,
                       xold1, xold2, scaled_objective_gradient, scaled_constraint_value,
                       scaled_constraint_gradient, low, upp, a0, a, c, d, move)
            xold2 = xold1.copy()
            xold1 = xval.copy()
            xval = xmma.copy()

            self.logged_values_data_frame.to_csv(self.__optimization_history_filepath,
                                                 float_format="%12.4e",
                                                 index=False)

            # Possibly output a restart file
            iteration_number += 1
            if (iteration_number % self.__restart_file_write_frequency) == 0:
                restart_filename = f"mma_restart_iteration_{iteration_number:04d}.npz"
                restart_filename = os.path.join(self.__optimization_restart_file_directory,
                                                f"mma_restart_iteration_{iteration_number:04d}.npz")
                np.savez(restart_filename, xval=xval, xold1=xold1, xold2=xold2, low=low, upp=upp,
                         objective_scale_factor=current_objective_scale_factor)


    ###################################################################################################################
    ###################################################################################################################
    def plot_history(self, figure_size_scale_factor: float = 2.5, save_figure: bool = True):
        """Plot the optimization history including all of the logged values.

        Args:
            figure_size_scale_factor: A float representating the scale factor on the figure size.
            save_figure: Save the figure in the output directory if True, otherwise do not save.

        Returns:
            None
        """
        if shutil.which("tex") and shutil.which("dvipng") and shutil.which("gs"):
            plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        if self.logged_values_data_frame is None:
            self.logged_values_data_frame = pd.read_csv(self.__optimization_history_filepath)

        def get_label(my_label):
            label_map = \
                {"Coarse Scale d_StressXX_d_StainXX": r"$L^{coarse}_{1111}$",
                 "Coarse Scale d_StressXX_d_StainYY": r"$L^{coarse}_{1122}$",
                 "Coarse Scale d_StressXX_d_StainXY": r"$L^{coarse}_{1112}$",
                 "Coarse Scale d_StressYY_d_StainYY": r"$L^{coarse}_{2222}$",
                 "Coarse Scale d_StressYY_d_StainXY": r"$L^{coarse}_{2212}$",
                 "Coarse Scale d_StressXY_d_StainXY": r"$L^{coarse}_{1212}$",
                 "Macroscale Compliance (objective)": r"Objective: Macroscale Compliance, $f_{ext} \cdot u$",
                 "Heaviside Projection Parameter": r"Smooth Projection Parameter, $\beta$",
                 "SIMP Exponent": r"SIMP Exponent $p$ in $\rho^p$",
                 "Volume Fraction (constraint)": r"Volume Fraction, $(\sum_{e=1}^{N_{elem}} \rho_e V_e ) / V_{total}$"}
            return label_map[my_label] if my_label in label_map else my_label

        font_size = 18
        def make_line_plot(dataframe=None, x=None, y_list=None, ax=None, color_list=None, style_list=None):
            for i, y in enumerate(y_list):
                ax.plot(dataframe[x], dataframe[y], color=color_list[i], linestyle=style_list[i],
                        linewidth=3.0, label=get_label(y))
            ax.set_xlabel(x, fontsize=font_size)
            ax.tick_params(axis='x', labelsize=font_size)
            ax.tick_params(axis='y', labelsize=font_size)
            ax.grid(True)
            ncols = 1 if len(y_list) < 3 else 2
            ax.legend(loc='best', prop={'size': font_size}, ncol=ncols)

        fig, my_axes = plt.subplots(nrows=2, ncols=2,
                                    figsize=(figure_size_scale_factor*6.4, figure_size_scale_factor*4.8))

        y_columns = ["Macroscale Compliance (objective)"]
        make_line_plot(dataframe=self.logged_values_data_frame, x="Optimization Iteration Number", y_list=y_columns,
                       ax=my_axes[0, 0], color_list=['b'], style_list=['solid'])

        y_columns = ["Volume Fraction (constraint)", "Constraint Upper Bound"]
        make_line_plot(dataframe=self.logged_values_data_frame, x="Optimization Iteration Number", y_list=y_columns,
                       ax=my_axes[0, 1], color_list=['b', 'r'], style_list=['solid', 'dashed'])
        lower_y_limit = round(max(0, np.amin(self.logged_values_data_frame["Volume Fraction (constraint)"]) - 0.1), 1)
        upper_y_limit = round(min(1, np.amax(self.logged_values_data_frame["Volume Fraction (constraint)"]) + 0.1), 1)
        my_axes[0, 1].set_ylim([lower_y_limit, upper_y_limit])
        yticks = np.linspace(lower_y_limit, upper_y_limit, 5)
        my_axes[0, 1].set_yticks(yticks)
        my_axes[0, 1].set_yticklabels([f"{yt:0.2f}" for yt in yticks])

        y_columns = ["Heaviside Projection Parameter", "SIMP Exponent"]
        make_line_plot(dataframe=self.logged_values_data_frame, x="Optimization Iteration Number", y_list=y_columns,
                       ax=my_axes[1, 0], color_list=['b', 'r'], style_list=['solid', 'dashed'])

        y_columns = ["Coarse Scale d_StressXX_d_StainXX","Coarse Scale d_StressXX_d_StainYY",
                     "Coarse Scale d_StressXX_d_StainXY","Coarse Scale d_StressYY_d_StainYY",
                     "Coarse Scale d_StressYY_d_StainXY","Coarse Scale d_StressXY_d_StainXY"]
        my_color_map = mpl.colormaps['rainbow'].resampled(len(y_columns))
        my_color_list = [my_color_map(i) for i in range(len(y_columns))]
        my_style_list = ['solid', 'solid', 'dashed', 'dashed', 'dotted', 'dotted']
        make_line_plot(dataframe=self.logged_values_data_frame, x="Optimization Iteration Number", y_list=y_columns,
                       ax=my_axes[1, 1], color_list=my_color_list, style_list=my_style_list)

        fig.tight_layout()
        if save_figure:
            figure_filepath = os.path.join(os.path.dirname(self.__optimization_history_filepath),
                                           'OptimizationHistory.pdf')
            fig.savefig(figure_filepath)
