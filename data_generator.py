import Surface_confined_inference as sci
from Surface_confined_inference.plot import plot_harmonics
from Surface_confined_inference.infer import get_input_parameters
import sys
import matplotlib.pyplot as plt
import numpy as np
psv = sci.SingleExperiment(
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
psv.boundaries = {"k0": [1e-3, 200], 
                    "E0_mean": [-0.1, 0.06],
                    "E0_std":[1e-5, 0.06],
                    "Cdl": [1e-5, 1e-3],
                    "gamma": [1e-11, 1e-9],
                    "Ru": [1, 1e3],
                    "CdlE1":[-1,1],
                    "CdlE2":[-0.01,0.01],
                    "CdlE3":[-0.001,0.001], 
                    "omega":[7, 13]   }
psv.fixed_parameters = {
    "alpha":0.5,
}
psv.dispersion_bins=[16]
psv.optim_list = ["E0_mean", "E0_std","k0", "Cdl", "gamma",  "Ru", "CdlE1", "CdlE2", "CdlE3", "omega"]


parameters=[0.03,0.04, 100, 1e-4, 1e-10, 100, 1e-2, -1e-4, 1e-6, 10]

psv.save_class("PSV_test.json")

ftv=sci.ChangeTechnique("PSV_test.json", "FTACV", 
                        input_parameters={
                            "E_start": -0.4,
                            "E_reverse": 0.3,
                            "v": 25e-3,
                            "delta_E":0.15,
                            "phase":0

                        })
dcv=sci.ChangeTechnique("PSV_test.json", "DCV", 
                        input_parameters={
                            "E_start": -0.4,
                            "E_reverse": 0.3,
                            "v": 25e-3,

                        })
fig, ax=plt.subplots(1,3)
sim_classes =[psv, ftv, dcv]
labels=["PSV", "FTV", "DCV"]
for i in range(0, len(sim_classes)):
    sim_class=sim_classes[i]
    axes=ax[i]

    nondim_t = sim_class.calculate_times(sampling_factor=200, dimensional=False)
    dim_t = sim_class.dim_t(nondim_t)
    current = sim_class.dim_i(sim_class.simulate(parameters,nondim_t))
    noisy_current=sci._utils.add_noise(current, 0.05*max(current))
    voltage=sim_class.get_voltage(dim_t, dimensional=True) 
    axes.plot(voltage, noisy_current)   
    axes.plot(voltage, current)
    with open("test_inference_{0}.txt".format(labels[i]), "w") as f:
        np.savetxt(f, np.column_stack((dim_t, noisy_current, voltage)))
    
plt.show()

