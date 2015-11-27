"""
  EQRM parameter file
All input files are first searched for in the input_dir,then in the
resources/data directory, which is part of EQRM.

All distances are in kilometers.
Acceleration values are in g.
Angles, latitude and longitude are in decimal degrees.

If a field is not used, set the value to None.


"""

from os.path import join, expanduser
from eqrm_code.parse_in_parameters import eqrm_data_home
from numpy import arange

# Path
working_path = join(expanduser("~"), 'Projects/scenario_Sydney')

# Operation Mode
#run_type = "hazard" 
run_type = "risk_mmi" 
is_scenario = True
#max_width = 15
site_tag = "sydney_soil"  # portfolio
#site_tag = "sydney" # grid
site_db_tag = ""
#return_periods = [10, 50, 100, 10000]
return_periods = [10, 50, 100, 200, 250, 474.56, 500, 974.78999999999996, 1000,
                  2474.9000000000001, 2500, 5000, 7500, 10000]
input_dir = join(working_path, 'input')
# ground motion field (rock, soil) of events
output_dir = join(working_path, 'scen_risk')
del working_path
use_site_indexes = False  # True
site_indexes = []
zone_source_tag = ""
event_control_tag = ""

# Scenario input
scenario_azimuth = 330
scenario_depth = 7.0
scenario_latitude = -33.914
scenario_longitude = 151.153
scenario_magnitude = 5.0 #5.4
scenario_dip = 40
scenario_number_of_events = 1

# Probabilistic input

# Attenuation
atten_models = ['Somerville09_Non_Cratonic', 'Akkar_2010_crustal', 'Campbell08']
atten_model_weights = [0.4, 0.2, 0.4]
atten_collapse_Sa_of_atten_models = True
atten_periods = [0.0, 0.3, 1.0]  # hyeuk
atten_threshold_distance = 400
atten_variability_method = 2  # None
atten_override_RSA_shape = None
atten_cutoff_max_spectral_displacement = False
atten_pga_scaling_cutoff = 2.0
atten_smooth_spectral_acceleration = None
atten_log_sigma_eq_weight = 0

# Amplification
use_amplification = True
amp_variability_method = None
amp_min_factor = 0.6
amp_max_factor = 10000

# Buildings

# Capacity Spectrum Method

# Vulnerability Input
vulnerability_variability_method = None

# Loss

# Save
save_hazard_map = False
save_total_financial_loss = False
save_building_loss = True
save_contents_loss = False
save_motion = False
save_prob_structural_damage = None

file_array = False

# If this file is executed the simulation will start.
# Delete all variables that are not EQRM attributes variables.
if __name__ == '__main__':
    from eqrm_code.analysis import main
    main(locals())