import numpy as np
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, femtoseconds, m_e, Å

#=========================================================================================================#
#First, we define the Hamiltonian of a single particle
#=========================================================================================================#

# interaction potential
def double_slit(particle):
     b = 2.0* Å # slits separation
     a = 0.5* Å # slits width
     d = 0.5* Å # slits depth

     return np.where( ((particle.x < - b/2 - a) | (particle.x > b/2 + a) | ((particle.x > -b/2)  
                      & (particle.x < b/2))) & ((particle.y < d/2) & (particle.y > -d/2) ),  1e5,  0)

def double_slit1(particle):
    b = 2.0 * Å  # slits separation
    a = 0.5 * Å  # slits width
    d = 0.5 * Å  # slits depth

    return np.where(((particle.x < -b/2 - a) | (particle.x > b/2 + a) | ((particle.x > -b/2)  
                      & (particle.x < b/2))) & ((particle.y < d/2) & (particle.y > -d/2)), 1e5, 0)

def double_slit2(particle):
    b = 2.0 * Å  # slits separation
    a = 0.5 * Å  # slits width
    d = 0.5 * Å  # slits depth
    shift_y = 5.0 * Å  # Shift of the second slit along the y-axis
    shift = 5.0 * Å  # Shift of the second slit along the x-axis
    return np.where(((particle.x < -b/2 - a + shift) | (particle.x > b/2 + a + shift) | ((particle.x > -b/2 + shift)  
                      & (particle.x < b/2 + shift))) & ((particle.y < d/2 + shift_y) & (particle.y > -d/2 + shift_y)), 1e5, 0)

def single_slit1(particle):
    a = 0.8 * Å  # slit width
    d = 0.5 * Å  # slit depth

    return np.where(((particle.x < -a/2) | (particle.x > a/2)) & ((particle.y < d/2) & (particle.y > -d/2)), 1e5, 0)

def single_slit2(particle):
    a = 3.0 * Å  # slit width
    d = 0.5 * Å  # slit depth
    shift_y = 2.0 * Å  # Shift of the slit along the y-axis
    shift_x = 3.0 * Å  # Shift of the slit along the x-axis

    return np.where(((particle.x < -a/2 + shift_x) | (particle.x > a/2 + shift_x)) & 
                    ((particle.y < d/2 + shift_y) & (particle.y > -d/2 + shift_y)), 1e5, 0)

def single_slit3(particle):
    a = 2.0 * Å  # slit width
    d = 0.5 * Å  # slit depth
    shift_y = 4.0 * Å  # Shift of the slit along the y-axis
    shift_x = -1.0 * Å  # Shift of the slit along the x-axis

    return np.where(((particle.x < -a/2 + shift_x) | (particle.x > a/2 + shift_x)) & 
                    ((particle.y < d/2 + shift_y) & (particle.y > -d/2 + shift_y)), 1e5, 0)

# def harmonic_potential(particle):
    m = m_e  # mass of the particle, in this case, electron mass
    omega = 0.1 * femtoseconds**-1  # angular frequency
    return 0.5 * m * omega**2 * (particle.x**2 + particle.y**2)

combined_potential = lambda particle: (
    single_slit1(particle) + 
    single_slit2(particle) + 
    single_slit3(particle) )

import matplotlib.pyplot as plt

# def extract_amplitude_at_y(sim, y_value):
#     time_steps = sim.times
#     x_grid = sim.grid.x
#     amplitude_profiles = []

#     for wavefunction in sim.wavefunctions:
#         # Interpolate the wavefunction values at y = y_value
#         amplitude_profile = np.interp(y_value, sim.grid.y, np.abs(wavefunction))
#         amplitude_profiles.append(amplitude_profile)

#     return time_steps, np.array(amplitude_profiles)

# def plot_amplitude_vs_time(time_steps, amplitude_profiles):
#     plt.figure(figsize=(10, 6))
#     plt.imshow(amplitude_profiles.T, aspect='auto', extent=[time_steps[0], time_steps[-1], -15, 15], origin='lower')
#     plt.colorbar(label='Amplitude')
#     plt.xlabel('Time')
#     plt.ylabel('x')
#     plt.title('Amplitude vs Time at y = 5')
#     plt.show()

# combined_potential = lambda particle: single_slit1(particle) + single_slit2(particle) + single_slit3(particle)
#combined_potential = lambda particle: np.maximum(single_slit1(particle), single_slit2(particle), single_slit3(particle))
H = Hamiltonian(particles=SingleParticle(m=m_e), 
                potential=combined_potential, 
                spatial_ndim=2, 
                N=216, 
                extent=30 * Å)

#=========================================================================================================#
# Define the wavefunction at t = 0  (initial condition)
#=========================================================================================================#

def initial_wavefunction(particle):
    #This wavefunction correspond to a gaussian wavepacket with a mean Y momentum equal to p_y0
    σ = 1.0 * Å
    v0 = 80 * Å / femtoseconds
    p_y0 = m_e * v0
    return np.exp( -1/(4* σ**2) * ((particle.x-0)**2+(particle.y+8* Å)**2)) / np.sqrt(2*np.pi* σ**2)  *np.exp(p_y0*particle.y*1j)


#=========================================================================================================#
# Set and run the simulation
#=========================================================================================================#


total_time = 0.7 * femtoseconds
sim = TimeSimulation(hamiltonian = H, method = "split-step")
sim.run(initial_wavefunction, total_time = total_time, dt = total_time/8000., store_steps = 800)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Run the simulation
# total_time = 0.7 * femtoseconds
# sim = TimeSimulation(hamiltonian=H, method="split-step")
# sim.run(initial_wavefunction, total_time=total_time, dt=total_time/8000., store_steps=800)

# # Extract and plot amplitude values at y = 5
# y_value = 5 * Å
# time_steps, amplitude_profiles = extract_amplitude_at_y(sim, y_value)
# plot_amplitude_vs_time(time_steps, amplitude_profiles)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#=========================================================================================================#
# Finally, we visualize the time dependent simulation
#=========================================================================================================#

visualization = init_visualization(sim)

visualization.animate(xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2, animation_duration = 10, fps = 30)

# import numpy as np
# import matplotlib.pyplot as plt
# from qmsolve import Å

# # Define a function to extract the amplitude at a specific y-coordinate
# def extract_amplitude_at_y(sim, y_value):
#     time_steps = sim.times
#     x_grid = sim.grid.x
#     amplitude_profiles = []

#     # Loop over each stored wavefunction in the simulation
#     for wavefunction in sim.wavefunctions:
#         # Interpolate the wavefunction values at y = y_value
#         amplitude_profile = np.interp(y_value, sim.grid.y, np.abs(wavefunction))
#         amplitude_profiles.append(amplitude_profile)

#     return time_steps, np.array(amplitude_profiles)

# # Define a function to plot the amplitude vs time
# def plot_amplitude_vs_time(time_steps, amplitude_profiles):
#     plt.figure(figsize=(10, 6))
#     plt.imshow(amplitude_profiles.T, aspect='auto', extent=[time_steps[0], time_steps[-1], -15, 15], origin='lower')
#     plt.colorbar(label='Amplitude')
#     plt.xlabel('Time (femtoseconds)')
#     plt.ylabel('x (Å)')
#     plt.title(f'Amplitude vs Time at y = {y_value/Å} Å')
#     plt.show()

# # Run the simulation (already done in your code)

# # Extract and plot amplitude values at y = 5 Å
# y_value = 5 * Å  # y-coordinate at which to extract the amplitude
# time_steps, amplitude_profiles = extract_amplitude_at_y(sim, y_value)
# plot_amplitude_vs_time(time_steps, amplitude_profiles)
