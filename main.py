import argparse
import hvlcalc
import auxfuncs
import typing
import scipy.optimize


class MainObj:
    def __init__(self, inputfile, outputdir, debug):
        self.config_file = auxfuncs.ConfigFile(inputfile, outputdir, debug)
        self.config_file.parse_config()
        self.hvl_calculator = hvlcalc.HVLCalculator(configfile=self.config_file)

    def calculate(self):
        self.hvl_calculator.calc_values()

    def write_results(self):
        outputfile = f'{self.config_file.config_general.output_direc}/' \
                     f'{self.config_file.config_general.folder_name}/output.txt'
        with open(outputfile, 'w') as f:
            f.write('--------HVLCalculator----------\n')
            f.write('--------Version Alpha----------\n')
            f.write('--------Results----------------\n')
            f.write('Unfiltered spectrum\n')
            hvlcalc.print_hvl_results(f, self.hvl_calculator.original_hvl, self.hvl_calculator.config_file)
            f.write('-------------------------------\n')
            if self.config_file.additional_filtration.filters:
                f.write('Filtered spectrum\n')
                hvlcalc.print_hvl_results(f, self.hvl_calculator.filtered_hvl, self.hvl_calculator.config_file)

    def write_spectrum_log(self, f: typing.TextIO)->None:
        f.write('-------------------------------\n')
        f.write('Spectrum Information\n')
        f.write('Energy\tProb\n')
        for i in range(len(self.config_file.spectrum.energy)):
            f.write(f'{self.config_file.spectrum.energy[i]}\t{self.config_file.spectrum.prob[i]}\n')
        f.write('-------------------------------\n')

    def write_coefficients_log(self, f: typing.TextIO, material: auxfuncs.Material)->None:
        f.write(f'Name:{material.name}\n')
        f.write(f'Density:{material.density}\n')
        if material.thickness>0:
            f.write(f'Thickness:{material.density}\n')
        if len(material.mu_en_rho) > 0:
            f.write(f'Energy\tMu/Rho\tMu_en/Rho\n')
        else:
            f.write(f'Energy\tMu/Rho\n')
        for i in range(len(material.energy)):
            if len(material.mu_en_rho) > 0:
                f.write(f'{material.energy[i]}\t{material.mu_rho[i]}\t{material.mu_en_rho[i]}\n')
            else:
                f.write(f'{material.energy[i]}\t{material.mu_rho[i]}\n')
        f.write('-------------------------------\n')


    def write_optimization_log(self, f: typing.TextIO, optimization: scipy.optimize.OptimizeResult):
        f.write(f'x: {optimization.x}\n')
        f.write(f'success: {optimization.success}\n')
        f.write(f'status: {optimization.status}\n')
        f.write(f'message: {optimization.message}\n')
        f.write(f'nit: {optimization.nit}\n')
        f.write(f'fun: {optimization.fun}\n')
        f.write('-------------------------------\n')

    def write_logs(self):
        logfile = f'{self.config_file.config_general.output_direc}/' \
                     f'{self.config_file.config_general.folder_name}/log.txt'
        with open(logfile, 'w') as f:
            self.write_spectrum_log(f)
            f.write('-------------------------------\n')
            f.write('Material Information\n')
            f.write('Air\n')
            self.write_coefficients_log(f, self.config_file.air_data)
            f.write('Reference Material\n')
            self.write_coefficients_log(f, self.config_file.reference_mat_data)
            if self.config_file.additional_filtration.filters:
                f.write('Additional Filtration\n')
                for material in self.config_file.additional_filtration.materials:
                    self.write_coefficients_log(f, material)
            f.write('Solver information\n')
            f.write('Unfiltered spectrum\n')
            f.write('HVL1\n')
            self.write_optimization_log(f, self.hvl_calculator.original_hvl.hvl1)
            f.write('HVL2\n')
            self.write_optimization_log(f, self.hvl_calculator.original_hvl.hvl2)
            f.write('Effective Energy\n')
            self.write_optimization_log(f, self.hvl_calculator.original_hvl.effective_energy)
            if self.config_file.additional_filtration.filters:
                self.write_optimization_log(f, self.hvl_calculator.filtered_hvl.hvl1)
                f.write('HVL2\n')
                self.write_optimization_log(f, self.hvl_calculator.filtered_hvl.hvl2)
                f.write('Effective Energy\n')
                self.write_optimization_log(f, self.hvl_calculator.filtered_hvl.effective_energy)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Algorithm to calculate the Half Value Layer values')
    parser.add_argument('-input', help='input configuration file')
    parser.add_argument('-output', help='output folder')
    parser.add_argument('--debug', help='debug mode', default='False', choices=['True', 'False'])

    args = parser.parse_args()
    inputfile = args.input
    outputdir = args.output
    debugmode = auxfuncs.check_bool(args.debug)

    calculator = MainObj(inputfile, outputdir, debugmode)
    calculator.calculate()
    calculator.write_results()
    calculator.write_logs()
    print('Done')

