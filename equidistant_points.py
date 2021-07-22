################################################################ Initialize ##########################################################
import hoomd
import hoomd.md
from hoomd.deprecated import dump
import numpy as np
from matplotlib import pyplot
from numpy import arange, pi, sin, cos, arccos
import matplotlib.tri as mtri
from scipy.spatial import Delaunay
import scipy.stats as st


hoomd.context.initialize("");


n_points = 100

R = 100; #radius of a sphere
R_on_sphere = 0.3; #radius of a particle on the sphere


N = n_points

snapshot = hoomd.data.make_snapshot(N=N,
									box=hoomd.data.boxdim(Lx=500, Ly=500, Lz=500),
									particle_types=['A'], 
									bond_types=['polymer']);



# num_pts = 1000
# x, y, z = [0], [0], [0]

#https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
indices = arange(0, n_points, dtype=float) + 0.5

phi = arccos(1 - 2*indices/n_points)
theta = pi * (1 + 5**0.5) * indices

x, y, z = list(R * cos(theta) * sin(phi)), list(R * sin(theta) * sin(phi)), list(R * cos(phi));

x = [0] + x
y = [0] + y
z = [0] + z

dist = np.sqrt((x[1] - x[2]) ** 2 + (y[1] - y[2]) ** 2 + (z[1] - z[2]) ** 2) # distance between points on the sphere


points3D = np.vstack([x, y, z]).T

tri = Delaunay(points3D)

snapshot.particles.position[:] = np.vstack(points3D[1:,:])
snapshot.particles.diameter[:] = np.array([2 * R_on_sphere] * n_points)
snapshot.particles.mass[:] = np.array([10e-5] * N) 

snapshot.particles.typeid[0:n_points] = 0

#bonds
faces = (np.sort(tri.simplices, axis=1) - \
				 np.ones(tri.simplices.shape))[:, 1:]# gives you vertices of each triangle
a = np.vstack((faces[:,0]).T)
b = np.vstack((faces[:,1]).T)
c = np.vstack((faces[:,2]).T)

arr_1d_1 = np.hstack((a, b))
arr_1d_2 = np.hstack((b, c))
arr_1d_3 = np.hstack((a, c))
arr_1d_4 = np.vstack((arr_1d_1, arr_1d_2, arr_1d_3))

snapshot.bonds.resize(arr_1d_4.shape[0]);
snapshot.bonds.group[:] = arr_1d_4 

system = hoomd.init.read_snapshot(snapshot);

############################################################### Define the forces ######################################################
nl = hoomd.md.nlist.cell();

# lj = hoomd.md.pair.lj(r_cut=10, nlist=nl);

# potential_type = 'WCA'

# beta = 0.95
sigma_AA = 10

epsilon_AA = 1.0

nl.reset_exclusions(exclusions = []);

r0 = dist*10
k = 0.1

fene = hoomd.md.bond.fene();
fene.bond_coeff.set('polymer', k=k, r0=r0, sigma=sigma_AA, epsilon= epsilon_AA);
# 'polymer', k=30.0, r0=1.5, sigma=1.0, epsilon= 2.0

# harmonic = hoomd.md.bond.harmonic();
# harmonic.bond_coeff.set('polymer', k=k, r0=r0);

############################################################# Select integrator #######################################################
hoomd.md.integrate.mode_standard(dt=0.0001);

all = hoomd.group.all();
integrator = hoomd.md.integrate.nvt(group=all, tau=1.0, kT=6);

integrator.randomize_velocities(seed=42)

####################################################### Write output ###################################################################
hoomd.analyze.log(filename="gabaidullina_drying_process_only_shell_sphericity_vs_number_of_vertices_fene_2.log",
				  quantities=['potential_energy', 'temperature'],
				  period=10,
				  overwrite=True);

hoomd.dump.gsd(filename="gabaidullina_drying_process_only_shell_sphericity_vs_number_of_vertices_fene_2.gsd", period=10, group=all, overwrite=True);

dump.pos(filename="gabaidullina_drying_process_only_shell_sphericity_vs_number_of_vertices_fene_2.pos", period=10)

#################################################### Run the simulation ###############################################################
sphericities = []
for t in range(10):

	hoomd.run(100)
	#calculate area and volume of the spherical confinement during the simulation (in dynamics)
	Area = 0
	Volume = 0
	
	# store tag -> index map
	indices = {}
	for i in range(N):
		p = system.particles[i]
		indices[p.tag] = i

	for face in faces:
		# for i in n_pointsgroupA: 
		# A = np.array(system.particles[int(face[0])].position) #current position of a particle
		# B = np.array(system.particles[int(face[1])].position)
		# C = np.array(system.particles[int(face[2])].position)
		A = np.array(system.particles[indices[int(face[0])]].position) #current position of a particle
		B = np.array(system.particles[indices[int(face[1])]].position)
		C = np.array(system.particles[indices[int(face[2])]].position)
		D = np.array([0, 0, 0])
		AB = B - A
		AC = C - A
		AD = D - A
		Area   += np.linalg.norm(np.cross(AB, AC)) / 2 #area of a triangle of the spherical shell
		Volume += np.linalg.norm(np.dot(np.cross(AB, AC), AD)) / 6 #volume of a triangle of the spherical shell

	sphericity = 36 * np.pi * Volume**2 / Area**3
	sphericities.append(sphericity)
	# if t % 10 == 0 : #in case you want to save each 10 steps
# with open("sphericity.npy", 'wb') as f: #save areas and volumes values in sphericity.npy file
# 	np.save(f, sphericities)

with open("sphericity-N={}-k={}_fene.npy".format(N, k), 'wb') as f: #save areas and volumes values in sphericity.npy file
	np.save(f, sphericities)
#################################################### Plot the graphs ##################################################################  
data = np.genfromtxt(fname="gabaidullina_drying_process_only_shell_sphericity_vs_number_of_vertices_fene_2.log", skip_header=True);

pyplot.figure(figsize=(4,2.2), dpi=140);
pyplot.plot(sphericities, 'o-'); #len(sphericity) ??? should be considered ???
pyplot.xlabel('time step');
pyplot.ylabel('sphericity');
pyplot.savefig('sphericity_vs_time_step_N={}-k={}_fene.pdf'.format(n_points, k), bbox_inches='tight');


