#docker pull multiphenics/multiphenics
#sudo docker run --shm-size=15g --cpus=8 -ti -v $(pwd):/home/fenics/shared multiphenics/multiphenics
import dolfin as df 
import matplotlib.pyplot as plt 
import multiphenics as mph 
import mshr
import time
from matplotlib import rc, rcParams

# plot parameters
plt.style.use('bmh') 
params = {'axes.labelsize': 28,
          'font.size': 22,
          'axes.titlesize': 28,
          'legend.fontsize': 20,
          'figure.titlesize': 26,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'figure.figsize':(10,8),          
          'legend.shadow': True,
          'patch.edgecolor': 'black'}
plt.rcParams.update(params)
# dolfin parameters
df.parameters["ghost_mode"] = "shared_facet" 
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters['allow_extrapolation'] = True
df.parameters["form_compiler"]["representation"] = 'uflacs'

# degree of interpolation for V and Vphi
degV = 2
degPhi = 2 + degV

# functions and parameters for elasticity
def sigma(u):
    return lambda_ * df.div(u)*df.Identity(2) + 2.0*mu*epsilon(u)

def epsilon(u):
    return (1.0/2.0)*(df.grad(u) + df.grad(u).T)

E = 2.0
nu = 0.3

lambda_ = E*nu/((1.0+nu)*(1.0-2.0*nu))
mu = E/(2.0*(1.0+nu))
print("lambda : ",lambda_)
print("mu : ",mu)

# Function used to write in the outputs files
def output_latex(f,A,B):
	for i in range(len(A)):
		f.write('(')
		f.write(str(A[i]))
		f.write(',')
		f.write(str(B[i]))
		f.write(')\n')
	f.write('\n')
 
"""
We define the level-sets function phi :
Here we consider the case of circle centered in (0.5,0.5) of radius sqrt(2)/4.
"""
class phi_expr(df.UserExpression) : 
    def eval(self, value, x):
        value[0] = -1.0/8.0 + (x[0]-0.5)**2 + (x[1]-0.5)**2 

    def value_shape(self):
        return (2,)



# We create the lists that we'll use to store errors and computation time for the phi-fem and standard fem
Time_assemble_dual_phi_fem, Time_solve_dual_phi_fem, Time_total_dual_phi_fem, error_l2_dual_phi_fem, error_h1_dual_phi_fem = [], [], [], [], []
Time_assemble_direct_phi_fem, Time_solve_direct_phi_fem, Time_total_direct_phi_fem, error_l2_direct_phi_fem, error_h1_direct_phi_fem = [], [], [], [], []
hh_phi_fem = []
Time_assemble_standard, Time_solve_standard, Time_total_standard, error_h1_standard, error_l2_standard,  hh_standard = [], [], [], [], [], []

###################################
# Computation of the standard FEM #
###################################  
# we compute the phi-fem for different sizes of cells
start,end,step = 1,6,1
for i in range(start,end,step): 
    print("Phi-fem iteration : ", i)
    # we define parameters and the "global" domain O
    H = 8*2**i
    background_mesh = df.UnitSquareMesh(H,H)
    
    # Creation of Omega_h
    V_phi = df.FunctionSpace(background_mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    Cell_omega = df.MeshFunction("size_t", background_mesh, background_mesh.topology().dim())
    Cell_omega.set_all(0)
    for cell in df.cells(background_mesh):  
        v1,v2,v3 = df.vertices(cell)
        if(phi(v1.point()) <= 0.0 or phi(v2.point()) <= 0.0 or phi(v3.point()) <= 0.0 or df.near(phi(v1.point()),0.0) or df.near(phi(v2.point()),0.0) or df.near(phi(v3.point()),0.0)):
            Cell_omega[cell] = 1
    mesh = df.SubMesh(background_mesh, Cell_omega, 1) 
    hh_phi_fem.append(mesh.hmax())   

    # Creation of the FunctionSpace for Phi on Omega_h
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    
    # Selection of cells and facets on the boundary for Omega_h^Gamma 
    mesh.init(1,2) 
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1) 
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())  
    cell_sub = df.MeshFunction("bool", mesh, mesh.topology().dim())
    facet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 1)
    vertices_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 2)
    Facet.set_all(0)
    Cell.set_all(0)
    cell_sub.set_all(0)
    facet_sub.set_all(0)
    vertices_sub.set_all(0)

    Dirichlet = 1
    for cell in df.cells(mesh) :
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if(phi(v1.point())*phi(v2.point()) <= 0.0 or df.near(phi(v1.point())*phi(v2.point()),0.0)) : 
                # Cells for dirichlet condition 
                Cell[cell] = 1
                cell_sub[cell] = 1
                for facett in df.facets(cell):  
                    Facet[facett] = 1
                    facet_sub[facett] = 1
                    v1, v2 = df.vertices(facett)
                    vertices_sub[v1], vertices_sub[v2] = 1,1


    File2 = df.File("sub.rtc.xml/mesh_function_2.xml")
    File2 << cell_sub
    File1 = df.File("sub.rtc.xml/mesh_function_1.xml")
    File1 << facet_sub
    File0 = df.File("sub.rtc.xml/mesh_function_0.xml")
    File0 << vertices_sub
    yp_res = mph.MeshRestriction(mesh,"sub.rtc.xml")
    
    ###################################
    # Computation of the dual phi FEM #
    ###################################  
    # Spaces and expressions of f, u_ex and boundary conditions
    V = df.VectorFunctionSpace(mesh, 'CG', degV, dim=2)
    u_ex = df.Expression(('sin(x[0]) * exp(x[1])', 'sin(x[1]) * exp(x[0])'), degree = 6, domain = mesh)
    f = - df.div(sigma(u_ex))   
    Z = df.TensorFunctionSpace(mesh,"CG",degV, shape = (2,2))
    Q_D = df.VectorFunctionSpace(mesh,"CG",degV, dim = 2)
    W = mph.BlockFunctionSpace([V,Z,Q_D], restrict=[None,yp_res,yp_res])
    uyp = mph.BlockTrialFunction(W)
    (u, y, p_D) = mph.block_split(uyp)
    vzq = mph.BlockTestFunction(W)
    (v, z, q_D) = mph.block_split(vzq)
    
    gamma_div, gamma_u, gamma_p, sigma_p, gamma_D, sigma_D = 1.0, 1.0, 1.0, 0.01, 20.0, 20.0
    h = df.CellDiameter(mesh)
    n = df.FacetNormal(mesh)
    u_D = u_ex * (1.0 + phi)
    
    # Modification of the measures to consider cells and facets on Omega_h^Gamma for the additional terms
    dx = df.Measure("dx", mesh, subdomain_data = Cell)
    ds = df.Measure("ds", mesh, subdomain_data = Facet)
    dS = df.Measure("dS", mesh, subdomain_data = Facet)


    # Construction of the bilinear and linear forms
    boundary_penalty = sigma_D*df.avg(h)*df.inner(df.jump(sigma(u),n), df.jump(sigma(v),n))*dS(1) \
                     + sigma_D*h**2*(df.inner(df.div(sigma(u)) ,df.div(sigma(v)) ))*dx(1)

    auv = df.inner(sigma(u), epsilon(v))*dx  \
        + boundary_penalty \
        + gamma_D*h**(-2)*df.inner(u,v)*dx(1) \
        - df.inner(df.dot(sigma(u),n),v)*ds
    auz = 0.0
    auq_D = - gamma_D*h**(-3)*df.dot(u,q_D*phi)*dx(1)
    
    ayv = 0.0
    ayz = 0.0
    ayq_D = 0.0
    
    ap_Dv = - gamma_D*h**(-3)*df.dot(v,p_D*phi)*dx(1)
    ap_Dz = 0.0
    ap_Dq_D = gamma_D*h**(-4)*df.inner(p_D*phi,q_D*phi)*dx(1)

    lv = df.inner(f,v)*dx  \
        - sigma_D*h**2*df.inner(f,  df.div(sigma(v)))*dx(1) \
        + gamma_D*h**(-2)*df.dot(u_D,v)*dx(1)
    lz = 0.0
    lq_D = - gamma_D*h**(-3)*df.inner(u_D,q_D*phi)*dx(1)

    a = [[auv,  auz,  auq_D],
         [ayv,  ayz,  ayq_D],
         [ap_Dv,ap_Dz,ap_Dq_D]]
    l = [lv,lz,lq_D]  
    
    # Resolution of the variationnal problem
    start_assemble = time.time()
    A = mph.block_assemble(a)
    B = mph.block_assemble(l)
    end_assemble = time.time()
    Time_assemble_dual_phi_fem.append(end_assemble-start_assemble)
    UU = mph.BlockFunction(W)
    start_solve = time.time()
    mph.block_solve(A, UU.block_vector(), B)
    end_solve = time.time()
    Time_solve_dual_phi_fem.append(end_solve-start_solve)
    Time_total_dual_phi_fem.append(Time_assemble_dual_phi_fem[-1] + Time_solve_dual_phi_fem[-1])
    u_h = UU[0]
    
    # Compute and store relative error for H1 and L2 norms
    relative_error_L2_dual_phi_fem = df.sqrt(df.assemble((df.inner(u_ex-u_h,u_ex-u_h)*df.dx)))/df.sqrt(df.assemble((df.inner(u_ex,u_ex))*df.dx))
    print("Relative error L2 dual phi FEM : ",relative_error_L2_dual_phi_fem)
    error_l2_dual_phi_fem.append(relative_error_L2_dual_phi_fem) 
    relative_error_H1_dual_phi_fem = df.sqrt(df.assemble((df.inner(df.grad(u_ex-u_h),df.grad(u_ex-u_h))*df.dx)))/df.sqrt(df.assemble((df.inner(df.grad(u_ex),df.grad(u_ex)))*df.dx))           
    error_h1_dual_phi_fem.append(relative_error_H1_dual_phi_fem) 
    print("Relative error H1 dual phi FEM : ",relative_error_H1_dual_phi_fem)

    #####################################
    # Computation of the direct phi FEM #
    #####################################  
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    boundary_penalty = sigma_D*df.avg(h)*df.inner(df.jump(sigma(phi*u),n), df.jump(sigma(phi*v),n))*dS(1) \
                     + sigma_D*h**2*(df.inner(df.div(sigma(phi*u)) ,df.div(sigma(phi*v)) ))*dx(1)
    a = df.inner(sigma(phi*u), epsilon(phi*v))*dx + boundary_penalty - df.inner(df.dot(sigma(phi*u),n),phi*v)*ds
    l = -df.inner(sigma(u_D), epsilon(phi*v))*dx + df.inner(f,phi*v)*dx  \
        - sigma_D*h**2*df.inner(f,  df.div(sigma(phi*v)))*dx(1) + df.inner(df.dot(sigma(u_D),n),phi*v)*ds- sigma_D*h**2*(df.inner(df.div(sigma(u_D)) ,df.div(sigma(phi*v)) ))*dx(1)
    
    # Resolution of the variationnal problem
    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(l)
    end_assemble = time.time()
    Time_assemble_direct_phi_fem.append(end_assemble-start_assemble)
    UU = df.Function(V)
    start_solve = time.time()
    df.solve(A, UU.vector(), B)
    end_solve = time.time()
    Time_solve_direct_phi_fem.append(end_solve-start_solve)
    Time_total_direct_phi_fem.append(Time_assemble_direct_phi_fem[-1] + Time_solve_direct_phi_fem[-1])
    u_h = UU*phi+ u_D
    
    # Compute and store relative error for H1 and L2 norms
    relative_error_L2_direct_phi_fem = df.sqrt(df.assemble((df.inner(u_ex-u_h,u_ex-u_h)*df.dx)))/df.sqrt(df.assemble((df.inner(u_ex,u_ex))*df.dx))
    print("Relative error L2 direct phi FEM : ",relative_error_L2_direct_phi_fem)
    error_l2_direct_phi_fem.append(relative_error_L2_direct_phi_fem) 
    relative_error_H1_direct_phi_fem = df.sqrt(df.assemble((df.inner(df.grad(u_ex-u_h),df.grad(u_ex-u_h))*df.dx)))/df.sqrt(df.assemble((df.inner(df.grad(u_ex),df.grad(u_ex)))*df.dx))           
    error_h1_direct_phi_fem.append(relative_error_H1_direct_phi_fem) 
    print("Relative error H1 direct phi FEM : ",relative_error_H1_direct_phi_fem)



###################################
# Computation of the standard FEM #
###################################     
domain = mshr.Circle(df.Point(0.5,0.5),df.sqrt(2.0)/4.0) # creation of the domain
for i in range(start, end, step):
    H = 8*2**(i-1) # to have approximately the same precision as in the phi-fem computation
    mesh = mshr.generate_mesh(domain,H)
    print("Standard fem iteration : ", i)
    # FunctionSpace P1
    V = df.VectorFunctionSpace(mesh, 'CG', degV, dim=2)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    n = df.FacetNormal(mesh)
    boundary = 'on_boundary'

    u_ex = df.Expression(('sin(x[0]) * exp(x[1])', 'sin(x[1]) * exp(x[0])'), degree = 6, domain = mesh)
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    f = - df.div(sigma(u_ex)) 
    
    # Boundary conditions
    u_D = u_ex * ( 1 + phi)
    bc = df.DirichletBC(V, u_D, boundary)

    # Variationnal problem
    a = df.inner(sigma(u), epsilon(v))*df.dx 
    L = df.dot(f,v)*df.dx
    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(L)
    end_assemble = time.time()
    Time_assemble_standard.append(end_assemble-start_assemble)
    u = df.Function(V)
    bc.apply(A,B) # apply Dirichlet boundary conditions to the problem    
    start_solve = time.time()
    df.solve(A, u.vector(), B)
    end_solve = time.time()
    Time_solve_standard.append(end_solve-start_solve)
    Time_total_standard.append(Time_assemble_standard[-1] + Time_solve_standard[-1])
 
    # Compute and store h and L2 H1 errors
    hh_standard.append(mesh.hmax())
    relative_error_L2_standard_fem = df.sqrt(df.assemble((df.inner(u_ex-u,u_ex-u)*df.dx)))/df.sqrt(df.assemble((df.inner(u_ex,u_ex))*df.dx))
    error_l2_standard.append(relative_error_L2_standard_fem)  
    print("Relative error L2 standard FEM : ",relative_error_L2_standard_fem)
    relative_error_H1_standard_fem = df.sqrt(df.assemble((df.inner(df.grad(u_ex-u),df.grad(u_ex-u))*df.dx)))/df.sqrt(df.assemble((df.inner(df.grad(u_ex),df.grad(u_ex)))*df.dx))
    error_h1_standard.append(relative_error_H1_standard_fem) 
    print("Relative error H1 standard FEM : ",relative_error_H1_standard_fem)

# Plot results : error/precision, Time/precision, Time/error and Total_time/error

"""plt.figure()
plt.loglog(hh_phi_fem,error_h1_dual_phi_fem,'o--', label=r'$\phi$-FEM $H^1$')
plt.loglog(hh_phi_fem,error_l2_dual_phi_fem,'o-', label=r'$\phi$-FEM $L^2$')
plt.loglog(hh_standard,error_h1_standard, '--x',label=r'Std FEM $H^1$')
plt.loglog(hh_standard,error_l2_standard, '-x',label=r'Std FEM $L^2$')
if degV == 1 :
    plt.loglog(hh_phi_fem, hh_phi_fem, '.', label="Linear")
    plt.loglog(hh_phi_fem,[hhh**2 for hhh in hh_phi_fem], '.',label="Quadratic")
elif degV == 2 :
    plt.loglog(hh_phi_fem,[hhh**2 for hhh in hh_phi_fem], '.',label="Quadratic")
    plt.loglog(hh_phi_fem,[hhh**3 for hhh in hh_phi_fem], '.',label="Cubic")

plt.xlabel("$h$")
plt.ylabel(r'$\frac{\|u-u_h\|}{\|u\|}$')
plt.legend(loc='upper right', ncol=2)
plt.title(r'Relative error : $ \frac{\|u-u_h\|}{\|u\|} $ for $L^2$ and $H^1$ norms', y=1.025)
plt.tight_layout()
plt.savefig('relative_error_Phi_Fem_Dirichlet_ElasticityP_{name0}.png'.format(name0=degV))
plt.show()

plt.figure()
plt.loglog(hh_phi_fem,Time_assemble_dual_phi_fem, '-o',label=r'Assemble $\phi$-FEM')
plt.loglog(hh_phi_fem,Time_solve_dual_phi_fem,'--o', label=r'Solve $\phi$-FEM')
plt.loglog(hh_standard,Time_assemble_standard, '-x',label=r'Assemble standard FEM')
plt.loglog(hh_standard,Time_solve_standard,'--x', label=r'Solve standard FEM')
plt.xlabel("$h$")
plt.ylabel("Time (s)")
plt.legend(loc='upper right')
plt.title("Computing time")
plt.tight_layout()
plt.savefig('Time_precision_P_{name0}.png'.format(name0=degV))
plt.show()
plt.figure()
plt.loglog(error_l2_dual_phi_fem,Time_assemble_dual_phi_fem, '-o',label=r'Assemble $\phi$-fem')
plt.loglog(error_l2_dual_phi_fem,Time_solve_dual_phi_fem,'--o', label=r'Solve $\phi$-fem')
plt.loglog(error_l2_standard,Time_assemble_standard,'-x', label="Assemble standard FEM")
plt.loglog(error_l2_standard,Time_solve_standard,'--x', label="Solve standard FEM")
plt.xlabel(r'$\frac{\|u-u_h\|_{L^2}}{\|u\|_{L^2}}$')
plt.ylabel("Time (s)")
plt.title(r'Computing time')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Time_error_P_{name0}.png'.format(name0=degV))
plt.show()
plt.figure()
plt.loglog(error_l2_dual_phi_fem,Time_total_dual_phi_fem,'-o', label=r'$\phi$-fem')
plt.loglog(error_l2_standard,Time_total_standard,'-x', label="Standard FEM")
plt.xlabel(r'$\frac{\|u-u_h\|_{L^2}}{\|u\|_{L^2}}$')
plt.ylabel("Time (s)")
plt.title(r'Computing time')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Total_time_error_Phi_Fem_Dirichlet_ElasticityP_{name0}.png'.format(name0=degV))
plt.show()"""


#  Write the output file for latex
f = open('output_Phi_Fem_Dirichlet_Elasticity_P{name0}.txt'.format(name0=degV),'w')
f.write('relative L2 norm dual phi fem: \n')	
output_latex(f, hh_phi_fem, error_l2_dual_phi_fem)
f.write('relative H1 norm dual phi fem : \n')	
output_latex(f, hh_phi_fem, error_h1_dual_phi_fem)
f.write('relative L2 norm and time dual  phi fem : \n')	
output_latex(f, error_l2_dual_phi_fem, Time_total_dual_phi_fem)
f.write('relative H1 norm and time dual phi fem : \n')	
output_latex(f, error_h1_dual_phi_fem, Time_total_dual_phi_fem)

f.write('relative L2 norm direct phi fem: \n')	
output_latex(f, hh_phi_fem, error_l2_direct_phi_fem)
f.write('relative H1 norm direct phi fem : \n')	
output_latex(f, hh_phi_fem, error_h1_direct_phi_fem)
f.write('relative L2 norm and time direct  phi fem : \n')	
output_latex(f, error_l2_direct_phi_fem, Time_total_direct_phi_fem)
f.write('relative H1 norm and time direct phi fem : \n')	
output_latex(f, error_h1_direct_phi_fem, Time_total_direct_phi_fem)

f.write('relative L2 norm classic fem: \n')	
output_latex(f, hh_standard, error_l2_standard)
f.write('relative H1 normclassic fem : \n')	
output_latex(f, hh_standard, error_h1_standard)
f.write('relative L2 norm and time classic fem : \n')	
output_latex(f, error_l2_standard, Time_total_standard)
f.write('relative H1 norm and time classic fem : \n')	
output_latex(f, error_h1_standard, Time_total_standard)
f.close()
