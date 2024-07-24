import Surface_confined_inference as sci
from Surface_confined_inference.plot import plot_harmonics
from Surface_confined_inference.infer import get_input_parameters
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
slurm_class = sci.SingleSlurmSetup(
     "PSV",
    {
        "Edc":-0.05,
        "omega": 10,
        "delta_E": 0.3,
        "area": 0.07,
        "Temp": 298,
        "N_elec": 1,
        "phase": 3*np.pi/2,
        "Surface_coverage": 1e-10,
        "num_peaks":15,
    },
)
slurm_class.boundaries = {"k0": [1e-3, 200], 
                    "E0_mean": [-0.1, 0.06],
                    "E0_std":[1e-5, 0.06],
                    "Cdl": [1e-5, 1e-3],
                    "gamma": [1e-11, 1e-9],
                    "Ru": [1, 1e3],
                    "CdlE1":[-1,1],
                    "CdlE2":[-0.01,0.01],
                    "CdlE3":[-0.001,0.001], 
                    "omega":[7, 13]   }
slurm_class.fixed_parameters = {
    "alpha":0.5,
}
slurm_class.dispersion_bins=[2]
slurm_class.optim_list = ["E0_mean", "E0_std","k0", "Cdl", "gamma",  "Ru", "CdlE1", "CdlE2", "CdlE3", "omega"]

slurm_class.setup(
    datafile="test_inference_PSV.txt",
    cpu_ram="8G",
    time="0-00:45:00",
    runs=5, 
    threshold=1e-8, 
    unchanged_iterations=200,   
    check_experiments={"DCV":"test_inference_DCV.txt", "FTACV":"test_inference_FTV.txt"},
    run=True,
)
