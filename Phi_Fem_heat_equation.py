#sudo docker run -ti -v $(pwd):/home/fenics/shared:z quay.io/fenicsproject/stable
from __future__ import print_function
import numpy as np
from dolfin import *
import sympy
import matplotlib.pyplot as plt
parameters['allow_extrapolation'] = True
import mshr

# Time of simulation
T=5.0

# Number of iterations
init_Iter = 0
Iter = 4

# parameter of the ghost penalty
sigma = 20

# Polynome Pk
polV = 1
degPhi = 2 + polV

# Ghost penalty
ghost = True

# plot the solution
Plot = False

# Compute the conditioning number
conditioning = False



# Function used to write in the outputs files
def output_latex(f,A,B):
	for i in range(len(A)):
		f.write('(')
		f.write(str(A[i]))
		f.write(',')
		f.write(str(B[i]))
		f.write(')\n')
	f.write('\n')


# Computation of the Exact solution and exact source term
t, x, y = sympy.symbols('tt xx yy')
u1 = sympy.exp(x)*sympy.sin(2*pi*y)*sympy.sin(t)
f1 =sympy.diff(u1, t) -sympy.diff(sympy.diff(u1, x),x)-sympy.diff(sympy.diff(u1, y),y)


# Initialistion of the output
size_mesh_phi_fem_vec = np.zeros(Iter)
error_L2_phi_fem_vec = np.zeros(Iter)
error_H1_phi_fem_vec = np.zeros(Iter)
cond_phi_fem_vec = np.zeros(Iter)
for i in range(init_Iter-1,Iter):
	print('###########################')
	print('## Iteration phi FEM ',i+1,'##')
	print('###########################')

	# Construction of the mesh
	N = int(10*2**((i)))
	mesh_macro = UnitSquareMesh(N,N)
	dt = mesh_macro.hmax()
	time = np.arange(0,T+dt,dt)
	V_phi = FunctionSpace(mesh_macro, "CG", degPhi)
	phi = Expression('-0.125+pow(x[0]-0.5,2)+pow(x[1]-0.5,2)',degree=degPhi,domain=mesh_macro)
	phi = interpolate(phi, V_phi)
	domains = MeshFunction("size_t", mesh_macro, mesh_macro.topology().dim())
	domains.set_all(0)
	for ind in range(mesh_macro.num_cells()):
		mycell = Cell(mesh_macro,ind)
		v1x,v1y,v2x,v2y,v3x,v3y = mycell.get_vertex_coordinates()
		if phi(v1x,v1y)<=0 or phi(v2x,v2y)<=0 or phi(v3x,v3y)<=0:
			domains[ind] = 1
	mesh = SubMesh(mesh_macro, domains, 1)
	V = FunctionSpace(mesh,'CG',polV)

	# Construction of phi
	V_phi = FunctionSpace(mesh, "CG", degPhi)
	phi = Expression('-0.125+pow(x[0]-0.5,2)+pow(x[1]-0.5,2)',degree=degPhi,domain=mesh)
	phi = interpolate(phi, V_phi)

	# Computation of the source term and solution
	f_expr = []
	for temps in time:
		f_expr = f_expr+ [Expression(sympy.ccode(f1).replace('xx', 'x[0]').replace('yy', 'x[1]').replace('tt', 'temps'),temps=temps,degree=polV+1,domain=mesh)]
	u_expr = []
	for temps in time:
		u_expr = u_expr + [Expression(sympy.ccode(u1).replace('xx', 'x[0]').replace('yy', 'x[1]').replace('tt', 'temps'),temps=temps,degree=4,domain=mesh)]

	# Facets and cells where we apply the ghost penalty
	mesh.init(1,2)
	facet_ghost = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
	cell_ghost = MeshFunction("size_t", mesh, mesh.topology().dim())
	facet_ghost.set_all(0)
	cell_ghost.set_all(0)
	for mycell in cells(mesh):
		for myfacet in facets(mycell):
			v1, v2 = vertices(myfacet)
			if phi(v1.point().x(),v1.point().y())*phi(v2.point().x(),v2.point().y())<0:
				cell_ghost[mycell] = 1
				for myfacet2 in facets(mycell):
					facet_ghost[myfacet2] = 1

	# Initialize cell function for domains
	dx = Measure("dx")(domain = mesh,subdomain_data = cell_ghost)
	ds = Measure("ds")(domain = mesh)
	dS = Measure("dS")(domain = mesh,subdomain_data = facet_ghost)

	# Resolution
	n = FacetNormal(mesh)
	h = CellDiameter(mesh)
	w = TrialFunction(V)
	v = TestFunction(V)
	u_n = Expression(sympy.ccode(u1).replace('xx', 'x[0]').replace('yy', 'x[1]').replace('tt', '0.0'),degree=polV+2,domain=mesh)
	sol = [u_n]
	a = dt**(-1)*w*v*phi**2*dx+inner(grad(phi*w),grad(phi*v))*dx - dot(inner(grad(phi*w),n),phi*v)*ds
	if ghost == True:
		a += sigma*avg(h)*dot(jump(grad(phi*w),n),jump(grad(phi*v),n))*dS(1)-sigma*h**2*inner(phi*w*dt**(-1)-div(grad(phi*w)),div(grad(phi*v)))*dx(1)
	for ind in range(1,len(time)):
		uD = u_expr[ind]*(1.0+phi)
		L = (dt**(-1)*u_n+f_expr[ind])*v*phi*dx
		L -= dt**(-1)*uD*v*phi*dx+inner(grad(uD),grad(phi*v))*dx - dot(inner(grad(uD),n),phi*v)*ds
		if ghost == True:
			L -= sigma*h**2*inner(f_expr[ind]+dt**(-1)*u_n,div(grad(phi*v)))*dx(1)
			L -= sigma*avg(h)*dot(jump(grad(uD),n),jump(grad(phi*v),n))*dS(1)+sigma*h**2*inner(div(grad(uD))-dt**(-1)*uD,div(grad(phi*v)))*dx(1)
		w_n1 = Function(V)
		solve(a == L, w_n1,solver_parameters={'linear_solver': 'mumps'})
		sol = sol + [w_n1*phi + uD]
		u_n = w_n1*phi +uD
		print('(',i+1,',',ind,'/',len(time)-1,')')

	# Computation of the error
	norm_L2_exact = 0.0
	err_L2 = 0.0
	norm_H1_exact = 0.0
	err_H1 = 0.0
	for j in range(len(time)):
		norm_L2_exact_j = assemble(u_expr[j]**2*dx(0))
		if norm_L2_exact < norm_L2_exact_j:
			norm_L2_exact = norm_L2_exact_j
		err_L2_j = assemble((sol[j]-u_expr[j])**2*dx(0))
		if err_L2 < err_L2_j:
			err_L2 = err_L2_j
		norm_H1_exact += assemble(dt*grad(u_expr[j])**2*dx(0))
		err_H1 += assemble(dt*grad(sol[j]-u_expr[j])**2*dx(0))
	err_L2 = err_L2**0.5/norm_L2_exact**0.5
	err_H1 = err_H1**0.5/norm_H1_exact**0.5
	size_mesh_phi_fem_vec[i] = mesh.hmax()
	error_L2_phi_fem_vec[i] = err_L2
	error_H1_phi_fem_vec[i] = err_H1
	print('h :',mesh.hmax())
	print('relative L2 error : ',err_L2)
	print('relative H1 error : ',err_H1)	
	if conditioning == True:
		A = np.matrix(assemble(a).array())
		ev, eV = np.linalg.eig(A)
		ev = abs(ev)
		cond = np.max(ev)/np.min(ev)
		cond_phi_fem_vec[i] = cond
		print("conditioning number x h^2",cond)
	print('')


# Initialistion of the output
size_mesh_standard_vec = np.zeros(Iter)
error_L2_standard_vec = np.zeros(Iter)
error_H1_standard_vec = np.zeros(Iter)
cond_standard_vec = np.zeros(Iter)
domain_mesh = mshr.Circle(Point(0.5,0.5),sqrt(2.0)/4.0) # creation of the domain
for i in range(init_Iter-1,Iter):
	print('###########################')
	print('## Iteration standard ',i+1,'##')
	print('###########################')

	# Construction of the mesh
	N = int(8*2**(i-1))
	mesh = mshr.generate_mesh(domain_mesh,N)
	dt = mesh.hmax()#10.0*mesh.hmax()**2
	time = np.arange(0,T+dt,dt)
	V = FunctionSpace(mesh,'CG',polV)

	# Construction of phi
	V_phi = FunctionSpace(mesh, "CG", degPhi)
	phi = Expression('-0.125+pow(x[0]-0.5,2)+pow(x[1]-0.5,2)',degree=degPhi,domain=mesh)
	phi = interpolate(phi, V_phi)

	# Computation of the source term
	f_expr = []
	for temps in time:
		f_expr = f_expr+ [Expression(sympy.ccode(f1).replace('xx', 'x[0]').replace('yy', 'x[1]').replace('tt', 'temps'),temps=temps,degree=polV+1,domain=mesh)]
	u_expr = []
	for temps in time:
		u_expr = u_expr + [Expression(sympy.ccode(u1).replace('xx', 'x[0]').replace('yy', 'x[1]').replace('tt', 'temps'),temps=temps,degree=4,domain=mesh)]

	# Initialize cell function for domains
	dx = Measure("dx")(domain = mesh)
	ds = Measure("ds")(domain = mesh)
	dS = Measure("dS")(domain = mesh)

	# Resolution
	n = FacetNormal(mesh)
	h = CellDiameter(mesh)
	u = TrialFunction(V)
	v = TestFunction(V)
	u_n = Expression(sympy.ccode(u1).replace('xx', 'x[0]').replace('yy', 'x[1]').replace('tt', '0.0'),degree=polV+2,domain=mesh)
	sol = [u_n]
	a = dt**(-1)*u*v*dx+inner(grad(u),grad(v))*dx 
	for ind in range(1,len(time)):
		uD = u_expr[ind]*(1.0+phi)
		bc = DirichletBC(V, uD,'on_boundary')
		L = dt**(-1)*u_n*v*dx+f_expr[ind]*v*dx
		u_n1 = Function(V)
		solve(a == L, u_n1,bc,solver_parameters={'linear_solver': 'mumps'})
		sol = sol + [u_n1]
		u_n = u_n1
		print('(',i+1,',',ind,'/',len(time)-1,')')


	# Computation of the error
	norm_L2_exact = 0.0
	err_L2 = 0.0
	norm_H1_exact = 0.0
	err_H1 = 0.0
	for j in range(len(time)):
		norm_L2_exact_j = assemble(u_expr[j]**2*dx)
		if norm_L2_exact < norm_L2_exact_j:
			norm_L2_exact = norm_L2_exact_j
		err_L2_j = assemble((sol[j]-u_expr[j])**2*dx)
		if err_L2 < err_L2_j:
			err_L2 = err_L2_j
		norm_H1_exact += assemble(dt*grad(u_expr[j])**2*dx)
		err_H1 += assemble(dt*grad(sol[j]-u_expr[j])**2*dx)
	err_L2 = err_L2**0.5/norm_L2_exact**0.5
	err_H1 = err_H1**0.5/norm_H1_exact**0.5
	size_mesh_standard_vec[i] = mesh.hmax()
	error_L2_standard_vec[i] = err_L2
	error_H1_standard_vec[i] = err_H1
	print('h :',mesh.hmax())
	print('relative L2 error : ',err_L2)
	print('relative H1 error : ',err_H1)	
	if conditioning == True:
		A = np.matrix(assemble(a).array())
		ev, eV = np.linalg.eig(A)
		ev = abs(ev)
		cond = np.max(ev)/np.min(ev)
		cond_standard_vec[i] = cond
		print("conditioning number x h^2",cond)
	print('')



# Print the output vectors  phi fem
print('Vector h phi fem:',size_mesh_phi_fem_vec)
print('Vector relative L2 error phi fem : ',error_L2_phi_fem_vec)
print('Vector relative H1 error phi fem : ',error_H1_phi_fem_vec)
print("conditioning number phi fem",cond_phi_fem_vec)

# Print the output vectors  standard
print('Vector h phi fem:',size_mesh_standard_vec)
print('Vector relative L2 error standard : ',error_L2_standard_vec)
print('Vector relative H1 error standard : ',error_H1_standard_vec)
print("conditioning number standard",cond_standard_vec)

#  Write the output file for latex
if ghost == False:
	f = open('outputs/output_heat_no_ghost.txt','w')
if ghost == True:
	f = open('outputs/output_heat_ghost.txt','w')
f.write('relative L2 norm phi fem : \n')	
output_latex(f,size_mesh_phi_fem_vec,error_L2_phi_fem_vec)
f.write('relative H1 norm phi fem : \n')	
output_latex(f,size_mesh_phi_fem_vec,error_H1_phi_fem_vec)
f.write('relative L2 norm standard : \n')	
output_latex(f,size_mesh_standard_vec,error_L2_standard_vec)
f.write('relative H1 norm standard : \n')	
output_latex(f,size_mesh_standard_vec,error_H1_standard_vec)
if conditioning == True:
	f.write('conditioning number phi fem  : \n')	
	output_latex(f,size_mesh_phi_fem_vec,cond_phi_fem_vec)
	f.write('conditioning number standard  : \n')	
	output_latex(f,size_mesh_standard_vec,cond_phi_fem_vec)
f.close()


