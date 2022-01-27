import numpy as np
import pandas as pd
from dataclasses import dataclass
import dataclasses
from typing import List, Tuple
import warnings
from pathlib import Path


def check_bool(value: str) -> bool:
    if value == 'True':
        return True
    elif value == 'False':
        return False
    else:
        raise ValueError('Values must be True or False')


def check_ascending(vec: list[float]):
    """Checks if a list is ordered in ascending order
    """
    for i in range(len(vec)-1):
        if vec[i+1] <= vec[i]:
            raise ValueError('Values in the list must be in ascending order')


def check_constant_step(vec: list[float]):
    """Checks if a list has a constant step
        """
    step = vec[1] - vec[0]
    for i in range(1, len(vec) - 1):
        if vec[i+1] - vec[i] != step:
            raise ValueError('Energy step must be constant')


def check_lower_than_zero(vec: list[float]):
    """Checks if a list has lower than zero values
    """
    for element in vec:
        if element < 0:
            raise ValueError('Values in the list must be greater or equal to zero')


def interp_log(val: list[float], x: list[float], y: list[float]) -> np.ndarray:
    """This function takes two arrays and perform a log-log linear interpolation

       :param val: The desired interpolated values
       :param x: x original values
       :param y: y original values
       :return: : out the interpolated values
       """
    x_log = np.log(x)
    y_log = np.log(y)
    val_log = np.log(val)
    out_log = np.interp(val_log, x_log, y_log)
    out = np.exp(out_log)
    return out


# parse config text in sections
def segment_section(text: str, section_name: str) -> str:
    """This function takes the configuration file string and selects only a specific section

    :param text: The configuration text
    :param section_name: The section name to be selected
    :return: text: The partitioned text with the desired section
    """
    init = f'[{section_name} CONFIG]'
    final = f'[END OF {section_name} CONFIG]'
    text = text.split(init)[1]
    text = text.split(final)[0]
    return text


# obtain parameter value in a section
def parse_parameter(text: str, parameter_name: str) -> str:
    """This function takes the section string and selects only a specific parameter

    :param text: the section string
    :param parameter_name: the name of the parameter to be selected
    :return: text: the string value of the desired parameter
    """
    init = f'{parameter_name} ='
    final = f'#'
    text = text.split(init)[1]
    text = text.split(final)[0]
    return text


# convert YES/NO string to boolean
def parse_bool(text: str, parameter_name: str) -> bool:
    """This function parses a string to bool values

    The function returns True or False based on YES/NO
    It raises an error otherwise

    :param text: the string to read
    :param parameter_name: the parameter name to search for
    :return: the bool value of the respective parameter
    """
    text = parse_parameter(text, parameter_name).strip().upper()
    if text == 'YES':
        return True
    elif text == 'NO':
        return False
    else:
        raise ValueError('String must be YES or NO')


def get_spectrum(spectrum_file: str, bin_mode: str = 'CENTER') -> Tuple[List[float], List[float]]:
    """ Read spectrum from file and transforms into a list

    It assumes that energy values are in the center of the bin

    :param spectrum_file: str: the file to read
    :param bin_mode: 'CENTER', 'LEFT' or 'RIGHT', where the bin location is considered
    :return: energy and prob: lists of floats of the energy and probability for each bin
    """
    print('Loading spectrum')
    spectrum = pd.read_csv(spectrum_file, sep='\s+|\t+|\s+\t+|\t+\s+', header=None, comment='#', engine='python').values
    spectrum = spectrum.T
    energy = spectrum[0]
    check_ascending(energy.tolist())
    check_lower_than_zero(energy.tolist())
    check_constant_step(energy.tolist())
    # converts left-bin to center-bin orientation
    if bin_mode.upper() == 'LEFT':
        spacing = (energy[1] - energy[0])/2
        energy += spacing
    elif bin_mode.upper() == 'RIGHT':
        spacing = (energy[1] - energy[0]) / 2
        energy -= spacing
    elif bin_mode.upper() == 'CENTER':
        spacing = 0
        energy += spacing
    else:
        raise ValueError('Bin mode must be LEFT, CENTER OR RIGHT')

    energy = energy.tolist()
    prob = spectrum[1].tolist()
    check_lower_than_zero(prob)
    return energy, prob


class ConfigFile:
    """This class handles the parameter parsing for the calculations

      """
    def __init__(self, config_loc: str, output_folder: str, debug: bool) -> None:
        """Initialization for ConfigFile class

        :param config_loc: str: the configuration file location
        :param output_folder: str: the output directory folder
        """
        self.config_general = ConfigGeneral()
        self.spectrum = SpectrumData()
        self.reference_mat_data = Material()
        self.air_data = Material()
        self.additional_filtration = MaterialList()
        self.config_general.input_file = config_loc
        self.config_general.output_direc = output_folder
        self.debug = debug

    def parse_general(self, text: str) -> None:
        """This method parse the configuration file text to extract general configurations

        :param text: the configuration text read
        """
        general_text = segment_section(text, 'GENERAL')
        folder_name = parse_parameter(general_text, 'FOLDER_NAME').strip()
        self.config_general.folder_name = folder_name
        spec_loc = parse_parameter(general_text, 'SPEC').strip()
        self.config_general.spec_loc = spec_loc
        bin_mode = parse_parameter(general_text, 'BIN_MODE').strip()
        self.config_general.bin_mode = bin_mode
        plot = parse_bool(general_text, 'PLOT')
        self.config_general.plot = plot
        air_table = parse_parameter(general_text, 'AIR_TABLE').strip()
        self.air_data.mu_data = air_table
        air_rho = float(parse_parameter(general_text, 'AIR_RHO'))
        self.air_data.density = air_rho
        ref_name = parse_parameter(general_text, 'REF_NAME').strip()
        self.reference_mat_data.name = ref_name
        al_table = parse_parameter(general_text, 'REF_TABLE').strip()
        self.reference_mat_data.mu_data = al_table
        al_rho = float(parse_parameter(general_text, 'REF_RHO'))
        self.reference_mat_data.density = al_rho
        return

    def parse_material(self, text: str) -> None:
        """This method parse the configuration file text to extract material configurations

        :param text: the configuration text read
        """
        material_text = segment_section(text, 'MATERIAL').splitlines()
        for line in material_text:
            line = line.split()
            if len(line) > 0:
                material = Material(name=line[0].strip(), density=float(line[1]), thickness=float(line[2]),
                                    mu_data=line[3].strip())
                self.additional_filtration.add_material(material)
        return

    def parse_config(self) -> None:
        """This method is a wrapper to unify all parsing methods and also read some additional files

        """
        with open(self.config_general.input_file, 'r') as file:
            text = file.read()
        self.parse_general(text)
        self.parse_material(text)
        self.read_mu()
        self.read_spectrum()
        self.create_folder()
        return

    # obtain attenuation coefficients
    def read_mu(self) -> None:
        """This method reads the attenuation coefficients for all materials and for the detector

        The energy absorption coefficient is optional for the materials, except if the air kerma calculation is
        necessary

        """
        print('Reading attenuation coefficients')
        for material in self.additional_filtration.materials:
            print(f'Reading for {material.name}')
            df = pd.read_csv(material.mu_data, comment='#', header=None, sep='\s+|\t+|\s+\t+|\t+\s+', engine='python')
            material.energy = df[0].values.tolist()
            material.energy_range = [np.min(material.energy), np.max(material.energy)]
            check_ascending(material.energy)
            check_lower_than_zero(material.energy)
            material.mu_rho = df[1].values.tolist()
            check_lower_than_zero(material.mu_rho)
            # if available we store the energy absorption coefficient
            if len(df.columns) > 2:
                material.mu_en_rho = df[2].values.tolist()
                check_lower_than_zero(material.mu_en_rho)
        # now we get for air and reference material
        print(f'Reading for air')
        df = pd.read_csv(self.air_data.mu_data, comment='#', header=None, sep='\s+|\t+|\s+\t+|\t+\s+',
                         engine='python')
        self.air_data.energy = df[0].values.tolist()
        check_ascending(self.air_data.energy)
        check_lower_than_zero(self.air_data.energy)
        self.air_data.mu_rho = df[1].values.tolist()
        check_lower_than_zero(self.air_data.mu_rho)
        self.air_data.mu_en_rho = df[2].values.tolist()
        check_lower_than_zero(self.air_data.mu_en_rho)
        self.air_data.name = 'air'

        print(f'Reading for {self.reference_mat_data.name}')

        df = pd.read_csv(self.reference_mat_data.mu_data, comment='#', header=None, sep='\s+|\t+|\s+\t+|\t+\s+',
                         engine='python')
        self.reference_mat_data.energy = df[0].values.tolist()
        check_ascending(self.reference_mat_data.energy)
        check_lower_than_zero(self.reference_mat_data.energy)
        self.reference_mat_data.mu_rho = df[1].values.tolist()
        check_lower_than_zero(self.reference_mat_data.mu_rho)
        # if available we store the energy absorption coefficient
        if len(df.columns) > 2:
            self.reference_mat_data.mu_en_rho = df[2].values.tolist()
            check_lower_than_zero(self.reference_mat_data.mu_en_rho)

        return

    def read_spectrum(self) -> None:
        """This method calls the function to read the spectrum and populates the energy and prob arrays

        """
        energy, prob = get_spectrum(self.config_general.spec_loc, self.config_general.bin_mode)
        self.spectrum.energy = energy
        self.spectrum.prob = prob
        return

    def check_energy_boundaries(self) -> None:
        """This method checks if the spectrum energy range is consistent with the energy grid of the materials

        """
        spec_min, spec_max = np.min(self.spectrum.energy), np.max(self.spectrum.energy)
        if spec_min < self.air_data.energy_range[0] or spec_max > self.air_data.energy_range[1]:
            warnings.warn('Energy range of spectrum is outside energy grid for air', UserWarning)
        if spec_min < self.reference_mat_data.energy_range[0] or spec_max > self.reference_mat_data.energy_range[1]:
            warnings.warn('Energy range of spectrum is outside energy grid for reference material', UserWarning)
        if self.additional_filtration.filters:
            for material in self.additional_filtration.materials:
                if spec_min < material.energy_range[0] or spec_max > material.energy_range[1]:
                    warnings.warn(f'Energy range of spectrum is outside energy grid for {material.name}', UserWarning)

    def create_folder(self) -> None:
        """This function creates the folder to store the results if the folder does not exist

        """
        Path(f'{self.config_general.output_direc}/{self.config_general.folder_name}').mkdir(parents=True,
                                                                                            exist_ok=True)





@dataclass
class ConfigGeneral:
    """This data class stores the general information

    """
    input_file: str = ''
    spec_loc: str = ''
    bin_mode: str = ''
    output_direc: str = ''
    folder_name: str = ''
    plot: bool = False


@dataclass
class SpectrumData:
    """This data class stores spectrum properties

    """
    energy: list[float] = dataclasses.field(default_factory=list)
    prob: list[float] = dataclasses.field(default_factory=list)


@dataclass
class Material:
    """This data class stores material properties

    """
    name: str = ''
    density: float = 0.0
    thickness: float = 0.0
    mu_data: str = ''
    energy: list[float] = dataclasses.field(default_factory=list)
    energy_range: list[float] = dataclasses.field(default_factory=list)
    mu_rho: list[float] = dataclasses.field(default_factory=list)
    mu_en_rho: list[float] = dataclasses.field(default_factory=list)


@dataclass
class MaterialList:
    """This data class stores different materials

    """
    materials: list[Material] = dataclasses.field(default_factory=list)
    filters: bool = False

    def add_material(self, material: Material) -> None:
        """This method adds a material to the list

        :param material: object from Material
        """
        self.materials.append(material)
        self.filters = True
        return

    def from_dict(self, mat_dict: dict) -> None:
        """This function read materials from a dictionary

        :param mat_dict:
        """
        mat_dict = mat_dict['materials']
        for entry in mat_dict:
            material = Material(name=entry['name'], density=entry['density'], thickness=entry['thickness'],
                                mu_data=entry['mu_data'],
                                energy=entry['energy'], mu_rho=entry['mu_rho'], mu_en_rho=entry['mu_en_rho'])
            self.add_material(material)
        return
