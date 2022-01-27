import numpy as np
import scipy.optimize
import auxfuncs
from scipy.optimize import minimize
from dataclasses import dataclass
import typing


def find_hvl(thick_ref: float, configfile: auxfuncs.ConfigFile,
             additional_filtration: bool = False, factor: float = 2.0, relative: bool = False) -> float:
    """Auxiliary function to determine the half value layer (HVL)

    :param thick_ref: The thickness in cm of the reference material
    :param configfile: ConfigFile object
    :param additional_filtration: if the spectrum is unfiltered or filtered
    :param factor: if =2, calculates the first HVL, if =4, calculates the second HVL
    :param relative: if the relative or absolute differences of HVL are computed
    :return: : the absolute difference between the original kerma/factor and the kerma calculated with thick_ref
    """

    kerma_0 = calc_kerma(0, configfile, additional_filtration)
    kerma_0 /= factor

    kerma_hvl = calc_kerma(thick_ref, configfile, additional_filtration)
    if relative:
        dif = (kerma_hvl - kerma_0)/kerma_0
    else:
        dif = np.abs(kerma_hvl - kerma_0)
    return dif


def debug_hvl(hvl_range: np.ndarray, configfile: auxfuncs.ConfigFile, additional_filtration: bool = False,
              factor: float = 2.0) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Auxiliary function to determine the half value layer (HVL)

    :param hvl_range: range to check the HVL values
    :param configfile: ConfigFile object
    :param additional_filtration: if the spectrum is unfiltered or filtered
    :param factor: if =2, calculates the first HVL, if =4, calculates the second HVL
    :return: : two numpy arrays
    """

    hvl_vec = np.linspace(hvl_range[0], hvl_range[1], 1000)
    dif = []
    for hvl in hvl_vec:
        dif.append(find_hvl(hvl, configfile,
                   additional_filtration, factor, True))
    dif = np.array(dif)
    return hvl_vec, dif


def calc_kerma(thick_ref: float, configfile: auxfuncs.ConfigFile,
               additional_filtration: bool = False) -> float:
    """Auxiliary to calculate the air kerma

     :param thick_ref: The thickness in cm of the reference material
     :param configfile: ConfigFile object
     :param additional_filtration: if the spectrum is unfiltered or filtered
     :return: : the absolute difference between the original kerma/factor and the kerma calculated with thick_ref
     """
    # air kerma formula:
    # K = Integral (E x PHI x mu_en/rho) dE
    # K = SUM (E X PHI X mu_en/rho x Delta)/ SUM(PHI x Delta)
    # since Delta is constant:
    # K = SUM (E x PHI x mu_en/rho)/ SUM (PHI)
    # where E is the energy in keV, PHI is the beam fluence and mu_en/rho the mass absorption coefficient of air
    # calculate mu_en_air
    energy = np.array(configfile.spectrum.energy)
    prob = np.array(configfile.spectrum.prob)

    air_data = configfile.air_data
    mu_en_air = auxfuncs.interp_log(energy.tolist(), air_data.energy, air_data.mu_en_rho)
    mu_en_air = mu_en_air*air_data.density

    at = calc_attenuation(thick_ref, configfile, additional_filtration)

    prob_new = prob*np.exp(-at)
    kerma = np.sum(mu_en_air*energy*prob_new)/np.sum(prob)

    return kerma


def calc_attenuation(thick_ref: float, configfile: auxfuncs.ConfigFile,
                     additional_filtration: bool = False) -> np.ndarray:
    """Calculate the beam attenuation with the additional filtration

    :param thick_ref: The thickness in cm of the reference material
    :param configfile: ConfigFile object
    :param additional_filtration: if the spectrum is unfiltered or filtered
    :return: : the sum of the attenuation coefficients
    """
    energy = np.array(configfile.spectrum.energy)
    at = np.zeros(len(energy))

    # calculate transmission due to reference material thickness
    mat_ref_data = configfile.reference_mat_data
    mu_mat_ref = auxfuncs.interp_log(energy.tolist(), mat_ref_data.energy, mat_ref_data.mu_rho)
    mu_mat_ref = mu_mat_ref * mat_ref_data.density

    at += mu_mat_ref*thick_ref
    if additional_filtration:
        for material in configfile.additional_filtration.materials:
            mu_mat = auxfuncs.interp_log(energy.tolist(), material.energy, material.mu_rho)
            mu_mat = mu_mat*material.density
            at += mu_mat*material.thickness

    return at


def calc_average_energy(configfile: auxfuncs.ConfigFile, additional_filtration: bool = False) -> float:
    """Calculate the beam attenuation with the additional filtration

    :param configfile: ConfigFile object
    :param additional_filtration: if the spectrum is unfiltered or filtered
    :return: : the average energy
    """
    at = calc_attenuation(0, configfile, additional_filtration)
    at = np.exp(-at)
    energy = configfile.spectrum.energy
    prob = np.array(configfile.spectrum.prob)*at
    average_energy = np.sum(energy * prob) / np.sum(prob)
    return average_energy


def find_effective_energy(energy: float, thick_ref: float, configfile: auxfuncs.ConfigFile, relative: bool = False):
    """Auxiliary function to determine the effective energy based on the reference material

    :param energy: The monoenergetic value to determine the HVL
    :param thick_ref: The thickness in cm of the reference material
    :param configfile: ConfigFile object
    :param relative: if absolute or relative differences are considered
    :return: : the difference between the calculated mu value and the expected one
    """
    mu_rho = np.log(2)/(thick_ref*configfile.reference_mat_data.density)
    mu_rho_interp = auxfuncs.interp_log([energy], configfile.reference_mat_data.energy,
                                        configfile.reference_mat_data.mu_rho)
    if not relative:
        dif = np.abs(mu_rho - mu_rho_interp)
    else:
        dif = 100*(mu_rho - mu_rho_interp)/mu_rho
    return dif


def debug_effective_energy(energy_range: np.ndarray, configfile: auxfuncs.ConfigFile, thick_ref: float
                           ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Auxiliary function to for debug effective energy

    :param energy_range: range to check the effective energy
    :param configfile: ConfigFile object
    :param thick_ref: thickness of the HVL reference material
    :return: : two numpy arrays
    """
    energy_vec = np.linspace(energy_range[0], energy_range[1], 1000)
    dif = []
    for energy in energy_vec:
        dif.append(find_effective_energy(energy, thick_ref, configfile,
                   True))
    dif = np.array(dif)
    return energy_vec, dif


@dataclass
class HVLValues:
    """This data class stores HVL results

    """
    hvl1: scipy.optimize.OptimizeResult = scipy.optimize.OptimizeResult
    hvl2: scipy.optimize.OptimizeResult = scipy.optimize.OptimizeResult
    effective_energy: scipy.optimize.OptimizeResult = scipy.optimize.OptimizeResult
    mean_energy: float = 0.0


class HVLCalculator:
    """This class handles all the HVL related calculations

     """
    def __init__(self, configfile: auxfuncs.ConfigFile):
        self.config_file = configfile
        self.original_hvl = HVLValues()
        self.filtered_hvl = HVLValues()
        self.method = 'Nelder-Mead'
        self.bounds_hvl = [(0.0, 100.0)]
        self.bounds_ef = [(np.min(configfile.reference_mat_data.energy), np.max(configfile.reference_mat_data.energy))]
        self.tol = 1E-6

    def calc_values(self):
        """Function to populate the HVLValues for the original and filtered spectrum

         """
        print('Calculating values')
        if self.config_file.additional_filtration.filters:
            hvl_vec = [self.original_hvl, self.filtered_hvl]
        else:
            hvl_vec = [self.original_hvl]
        additional_filter = [False, True]
        for idx, hvl_values in enumerate(hvl_vec):
            hvl_values.hvl1 = self.calc_hvl(additional_filter=additional_filter[idx], factor=2.0)
            hvl_values.hvl2 = self.calc_hvl(additional_filter=additional_filter[idx], factor=4.0)
            hvl_values.effective_energy = self.calc_effective_energy(hvl_values.hvl1.x)
            hvl_values.mean_energy = calc_average_energy(self.config_file, additional_filtration=additional_filter[idx])
        if self.config_file.config_general.plot:
            self.plot()
        if self.config_file.debug:
            self.plot_debug()

    def calc_hvl(self, additional_filter: bool, factor: float) -> scipy.optimize.OptimizeResult:
        """Wrapper to call the sympy minimize function to determine the half value layer

        :param additional_filter: if the spectrum is unfiltered or filtered
        :param factor: if =2, calculates the first HVL, if =4, calculates the second HVL
        :return: : the results of the minimize function
        """
        hvl_val = minimize(find_hvl, 0.0, args=(self.config_file, additional_filter, factor),
                           method=self.method, bounds=self.bounds_hvl, tol=self.tol)
        return hvl_val

    def calc_effective_energy(self, hvl1: float) -> scipy.optimize.OptimizeResult:
        """Wrapper to call the sympy minimize function to determine the effective energy

        :param hvl1: the first calculated half value layer
        :return: : the results of the minimize function
        """
        init_guess = (self.bounds_ef[0][0] + self.bounds_ef[0][1]) / 2
        effective_energy = minimize(find_effective_energy, init_guess,
                                    args=(hvl1, self.config_file), method=self.method,
                                    bounds=self.bounds_ef, tol=self.tol)
        return effective_energy

    def plot(self) -> None:
        """Function to generate plots
        """
        print('Generating plots')
        import matplotlib.pyplot as plt
        plt.figure()
        plt.step(self.config_file.spectrum.energy,
                 self.config_file.spectrum.prob/np.sum(self.config_file.spectrum.prob), where='mid', label='Original')
        ymin = 0.0
        ymax = np.max(self.config_file.spectrum.prob/np.sum(self.config_file.spectrum.prob))*1.01
        if self.config_file.additional_filtration.filters:
            at = calc_attenuation(0.0, self.config_file, self.config_file.additional_filtration.filters)
            prob = self.config_file.spectrum.prob
            prob_new = prob * np.exp(-at)
            plt.step(self.config_file.spectrum.energy, prob_new / np.sum(prob_new), where='mid',
                     label='Filtered')
            plt.text(0.05, 0.3, f'Transmission = {np.round(np.sum(prob_new) / np.sum(prob), 3)}',
                     transform=plt.gca().transAxes)
            if np.max(prob_new / np.sum(prob_new))*1.01 > ymax:
                ymax = np.max(prob_new / np.sum(prob_new))*1.01
        plt.xlabel('Energy (eV)')
        plt.ylabel('Relative fluence')
        plt.legend(loc='best')
        plt.xlim((np.min(self.config_file.spectrum.energy), np.max(self.config_file.spectrum.energy)))
        plt.ylim((ymin, ymax))
        plt.tight_layout()
        plt.savefig(f'{self.config_file.config_general.output_direc}/'
                    f'{self.config_file.config_general.folder_name}/plot.png', dpi=200)

    def plot_debug(self) -> None:
        """Function to generate debug plots
        """
        print('Generating debug plots')
        import matplotlib.pyplot as plt
        plt.figure()

        hvl_interval = np.array([self.original_hvl.hvl1.x[0]/4, self.original_hvl.hvl1.x[0]*4])

        x, y = debug_hvl(hvl_interval, self.config_file, False, 2.0)
        plt.plot(x, y, label='Original', color='blue')
        plt.axvline(x=self.original_hvl.hvl1.x[0], color='blue', ls='--')
        if self.config_file.additional_filtration.filters:
            hvl_interval = np.array([self.filtered_hvl.hvl1.x[0]/4, self.filtered_hvl.hvl1.x[0]*4])
            x, y = debug_hvl(hvl_interval, self.config_file, True, 2.0)
            plt.plot(x, y, label='Filtered', color='red')
            plt.axvline(x=self.filtered_hvl.hvl1.x[0], color='red', ls='--')
        plt.xlabel('HVL (mmAl)')
        plt.ylabel('Error (%)')
        plt.legend(loc='best')
        plt.axhline(ls='--', color='gray')
        plt.tight_layout()
        plt.savefig(f'{self.config_file.config_general.output_direc}/'
                    f'{self.config_file.config_general.folder_name}/debug_hvl_plot.png', dpi=200)

        plt.figure()
        energy_range = np.array([self.original_hvl.effective_energy.x[0]/1.5,
                                      self.original_hvl.effective_energy.x[0]*1.5])
        x, y = debug_effective_energy(energy_range,  self.config_file, self.original_hvl.hvl1.x[0])
        plt.plot(x, y, label='Original', color='blue')
        plt.axvline(x=self.original_hvl.effective_energy.x[0], color='blue', ls='--')
        if self.config_file.additional_filtration.filters:
            energy_range = np.array([self.filtered_hvl.effective_energy.x[0]/1.5,
                                     self.filtered_hvl.effective_energy.x[0]*1.5])
            x, y = debug_effective_energy(energy_range,
                                          self.config_file, self.filtered_hvl.hvl1.x[0])
            plt.plot(x, y, label='Filtered', color='red')
            plt.axvline(x=self.filtered_hvl.effective_energy.x[0], color='red', ls='--')
        plt.xlabel('Effective Energy (eV)')
        plt.ylabel('Error (%)')
        plt.legend(loc='best')
        plt.axhline(ls='--', color='gray')
        plt.tight_layout()
        plt.savefig(f'{self.config_file.config_general.output_direc}/'
                    f'{self.config_file.config_general.folder_name}/debug_effective_energy_plot.png', dpi=200)


def print_hvl_results(file: typing.TextIO, hvl: HVLValues, config: auxfuncs.ConfigFile):
    file.write(f'HVL1 (mm of {config.reference_mat_data.name}): {round(hvl.hvl1.x[0]*10,3)}\n')
    file.write(f'HVL2 (mm of {config.reference_mat_data.name}): {round(hvl.hvl2.x[0] * 10, 3)}\n')
    file.write(f'Average Energy (eV): {round(hvl.mean_energy, 1)}\n')
    file.write(f'Effective Energy (eV): {round(hvl.effective_energy.x[0], 1)}\n')
