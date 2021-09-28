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

lambda_ = 1.25
mu = 1.0
rho = 1.0

E = 2.0
nu = 0.3

lambda_ = E*nu/((1.0+nu)*(1.0-2.0*nu))
mu = E/(2.0*(1.0+nu))
print(lambda_)
print(mu)

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

# Dirichlet and Neumann boundaries    
def dirichlet(point):
    return point.x() > 0.5 - df.DOLFIN_EPS

def neumann(point):
    return point.x() < 0.5 + df.DOLFIN_EPS

def dirichlet_inter_neumann(point):
    return df.Near(point.x(), 0.5)


# We create the lists that we'll use to store errors and computation time for the phi-fem and standard fem
Time_assemble_phi, Time_solve_phi, Time_total_phi, error_l2_phi, error_h1_phi, hh_phi = [], [], [], [], [], []
Time_assemble_standard, Time_solve_standard, Time_total_standard, error_h1_standard, error_l2_standard,  hh_standard = [], [], [], [], [], []

# we compute the phi-fem for different sizes of cells
start,end,step = 1,3,1
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
    hh_phi.append(mesh.hmax())   

    # Creation of the FunctionSpace for Phi on Omega_h
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    
    # Selection of cells and facets on the boundary for Omega_h^Gamma 
    mesh.init(1,2) 
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1) 
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())  
    cell_dirichlet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim())
    facet_dirichlet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 1)
    vertices_dirichlet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 2)
    cell_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim())
    facet_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 1)
    vertices_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 2)
    Facet.set_all(0)
    Cell.set_all(0)
    cell_dirichlet_sub.set_all(0)
    facet_dirichlet_sub.set_all(0)
    vertices_dirichlet_sub.set_all(0)
    cell_neumann_sub.set_all(0)
    facet_neumann_sub.set_all(0)
    vertices_neumann_sub.set_all(0)

    Neumann, Dirichlet, Interface = 1, 2, 3
    for cell in df.cells(mesh) :
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if(phi(v1.point())*phi(v2.point()) <= 0.0 or df.near(phi(v1.point())*phi(v2.point()),0.0)) : 
                # check if the cell is a cell for Dirichlet condition or Neumann condition and add every cells, facets, vertices to the restricition
                Cell[cell] = Interface
                for facett in df.facets(cell):  
                    Facet[facett] = Interface
                vc1,vc2,vc3 = df.vertices(cell) 
                # Cells for dirichlet condition
                if dirichlet(vc1.point()) and dirichlet(vc2.point()) and dirichlet(vc3.point()): 
                    Cell[cell] = Dirichlet
                    cell_dirichlet_sub[cell] = 1
                    vertices_dirichlet_sub[vc1], vertices_dirichlet_sub[vc2], vertices_dirichlet_sub[vc3] = 1,1,1
                    for facett in df.facets(cell):  
                        Facet[facett] = Dirichlet
                        facet_dirichlet_sub[facett] = 1

                # Cells for Neumann condition
                if neumann(vc1.point()) and neumann(vc2.point()) and neumann(vc3.point()): 
                    Cell[cell] = Neumann
                    cell_neumann_sub[cell] = 1
                    vertices_neumann_sub[vc1], vertices_neumann_sub[vc2], vertices_neumann_sub[vc3] = 1,1,1
                    for facett in df.facets(cell):  
                        Facet[facett] = Neumann
                        facet_neumann_sub[facett] = 1

    File2 = df.File("sub_dirichlet.rtc.xml/mesh_function_2.xml")
    File2 << cell_dirichlet_sub
    File1 = df.File("sub_dirichlet.rtc.xml/mesh_function_1.xml")
    File1 << facet_dirichlet_sub
    File0 = df.File("sub_dirichlet.rtc.xml/mesh_function_0.xml")
    File0 << vertices_dirichlet_sub
    yp_dirichlet_res = mph.MeshRestriction(mesh,"sub_dirichlet.rtc.xml")
    File2 = df.File("sub_neumann.rtc.xml/mesh_function_2.xml")
    File2 << cell_neumann_sub
    File1 = df.File("sub_neumann.rtc.xml/mesh_function_1.xml")
    File1 << facet_neumann_sub
    File0 = df.File("sub_neumann.rtc.xml/mesh_function_0.xml")
    File0 << vertices_neumann_sub
    yp_neumann_res = mph.MeshRestriction(mesh,"sub_neumann.rtc.xml")
    
    # Spaces and expressions of f, u_ex and boundary conditions
    V = df.VectorFunctionSpace(mesh, 'CG', degV, dim=2)
    u_ex = df.Expression(('sin(x[0]) * exp(x[1])', 'sin(x[1]) * exp(x[0])'), degree = 6, domain = mesh)
    f = - df.div(sigma(u_ex))   
    Z_N = df.TensorFunctionSpace(mesh,"CG",degV, shape = (2,2))
    Q_N = df.VectorFunctionSpace(mesh,"DG",degV-1, dim = 2)
    Q_D = df.VectorFunctionSpace(mesh,"CG",degV, dim = 2)
    W = mph.BlockFunctionSpace([V,Z_N,Q_N,Q_D], restrict=[None,yp_neumann_res,yp_neumann_res,yp_dirichlet_res])
    uyp = mph.BlockTrialFunction(W)
    (u, y, p_N, p_D) = mph.block_split(uyp)
    vzq = mph.BlockTestFunction(W)
    (v, z, q_N, q_D) = mph.block_split(vzq)
    
    gamma_div, gamma_u, gamma_p, sigma_p, gamma_D, sigma_D = 1.0, 1.0, 1.0, 0.01, 20.0, 20.0
    h = df.CellDiameter(mesh)
    n = df.FacetNormal(mesh)
    g = df.dot(sigma(u_ex),df.grad(phi))/(df.inner(df.grad(phi),df.grad(phi))**0.5) \
        + u_ex*phi
    u_D = u_ex * (1.0 + phi)
    
    # Modification of the measures to consider cells and facets on Omega_h^Gamma for the additional terms
    dx = df.Measure("dx", mesh, subdomain_data = Cell)
    ds = df.Measure("ds", mesh, subdomain_data = Facet)
    dS = df.Measure("dS", mesh, subdomain_data = Facet)


    # Construction of the bilinear and linear forms
    boundary_penalty = sigma_p*df.avg(h)*df.inner(df.jump(sigma(u),n), df.jump(sigma(v),n))*dS(Neumann) \
                     + sigma_D*df.avg(h)*df.inner(df.jump(sigma(u),n), df.jump(sigma(v),n))*(dS(Dirichlet)+dS(Interface)) \
                     + sigma_D*h**2*(df.inner(df.div(sigma(u)) ,df.div(sigma(v)) ))*dx(Dirichlet)
    
    phi_abs = df.inner(df.grad(phi),df.grad(phi))**0.5

    auv = df.inner(sigma(u), epsilon(v))*dx  \
        + gamma_u*df.inner(sigma(u),sigma(v))*dx(Neumann) \
        + boundary_penalty \
        + gamma_D*h**(-2)*df.inner(u,v)*dx(Dirichlet) \
        - df.inner(df.dot(sigma(u),n),v)*(ds(Dirichlet) +ds(Interface))

    auz = gamma_u*df.inner(sigma(u),z)*dx(Neumann) 
    auq_N = 0.0  
    auq_D = - gamma_D*h**(-3)*df.dot(u,q_D*phi)*dx(Dirichlet)
    
    ayv = df.inner(df.dot(y,n),v)*ds(Neumann) + gamma_u*df.inner(y,sigma(v))*dx(Neumann) 
    ayz = gamma_u*df.inner(y,z)*dx(Neumann) + gamma_div*df.inner(df.div(y), df.div(z))*dx(Neumann) \
        + gamma_p*h**(-2)*df.inner(df.dot(y,df.grad(phi)), df.dot(z,df.grad(phi)))*dx(Neumann)
    ayq_N = gamma_p*h**(-3)*df.inner(df.dot(y,df.grad(phi)), q_N*phi)*dx(Neumann)
    ayq_D = 0.0
    
    ap_Nv =  0.0
    ap_Nz = gamma_p*h**(-3)*df.inner(p_N*phi, df.dot(z,df.grad(phi)))*dx(Neumann)
    ap_Nq_N = gamma_p*h**(-4)*df.inner(p_N*phi,q_N*phi)*dx(Neumann) 
    ap_Nq_D = 0.0
    
    ap_Dv = - gamma_D*h**(-3)*df.dot(v,p_D*phi)*dx(Dirichlet)
    ap_Dz = 0.0
    ap_Dq_N = 0.0
    ap_Dq_D = gamma_D*h**(-4)*df.inner(p_D*phi,q_D*phi)*dx(Dirichlet)

    lv = df.inner(f,v)*dx  \
        + sigma_D*h**2*df.inner(f, - df.div(sigma(v)))*dx(Dirichlet) \
        + gamma_D*h**(-2)*df.dot(u_D,v)*dx(Dirichlet)
    lz = gamma_div * df.inner(f, df.div(z))*dx(Neumann) - gamma_p*h**(-2)*df.inner(g*phi_abs, df.dot(z,df.grad(phi)))*dx(Neumann)
    lq_N = - gamma_p*h**(-3)*df.inner(g*phi_abs,q_N*phi)*dx(Neumann)
    lq_D = - gamma_D*h**(-3)*df.inner(u_D,q_D*phi)*dx(Dirichlet)

    a = [[auv,  auz,  auq_N,  auq_D],
         [ayv,  ayz,  ayq_N,  ayq_D],
         [ap_Nv,ap_Nz,ap_Nq_N,ap_Nq_D], 
         [ap_Dv,ap_Dz,ap_Dq_N,ap_Dq_D]]
    l = [lv,lz,lq_N,lq_D]  
    
    # Resolution of the variationnal problem
    start_assemble = time.time()
    A = mph.block_assemble(a)
    B = mph.block_assemble(l)
    end_assemble = time.time()
    Time_assemble_phi.append(end_assemble-start_assemble)
    UU = mph.BlockFunction(W)
    start_solve = time.time()
    mph.block_solve(A, UU.block_vector(), B)
    end_solve = time.time()
    Time_solve_phi.append(end_solve-start_solve)
    Time_total_phi.append(Time_assemble_phi[-1] + Time_solve_phi[-1])
    u_h = UU[0]
    

    # Compute and store relative error for H1 and L2 norms
    relative_error_L2_phi_fem = df.sqrt(df.assemble((df.inner(u_ex-u_h,u_ex-u_h)*df.dx)))/df.sqrt(df.assemble((df.inner(u_ex,u_ex))*df.dx))
    print("Relative error L2 phi FEM : ",relative_error_L2_phi_fem)
    error_l2_phi.append(relative_error_L2_phi_fem) 
    relative_error_H1_phi_fem = df.sqrt(df.assemble((df.inner(df.grad(u_ex-u_h),df.grad(u_ex-u_h))*df.dx)))/df.sqrt(df.assemble((df.inner(df.grad(u_ex),df.grad(u_ex)))*df.dx))           
    error_h1_phi.append(relative_error_H1_phi_fem) 
    print("Relative error H1 phi FEM : ",relative_error_H1_phi_fem)


# Computation of the standard FEM       
domain = mshr.Circle(df.Point(0.5,0.5),df.sqrt(2.0)/4.0) # creation of the domain
domain.set_subdomain(1, mshr.Rectangle(df.Point(0.0, 0.0), df.Point(0.5, 1.0))) 
for i in range(start, end, step):
    H = 8*2**(i-1) # to have approximately the same precision as in the phi-fem computation
    mesh = mshr.generate_mesh(domain,H)
    print("Standard fem iteration : ", i)
    # FunctionSpace P1
    V = df.VectorFunctionSpace(mesh, 'CG', degV, dim=2)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    n = df.FacetNormal(mesh)
    boundary = 'on_boundary && x[0] >= 0.5'

    u_ex = df.Expression(('sin(x[0]) * exp(x[1])', 'sin(x[1]) * exp(x[0])'), degree = 6, domain = mesh)
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    f = - df.div(sigma(u_ex)) 
    
    # Boundary conditions
    u_D = u_ex * ( 1 + phi)
    bc = df.DirichletBC(V, u_D, boundary)
    g = df.dot(sigma(u_ex),df.grad(phi))/(df.inner(df.grad(phi),df.grad(phi))**0.5) \
        + u_ex*phi

    # selection facet
    mesh.init(1,2) 
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1) 
    Facet.set_all(0)
    Neumann, Dirichlet = 1, 2
    for facet in df.facets(mesh) :
        v1,v2 = df.vertices(facet) 
        # Cells for dirichlet condition
        if dirichlet(v1.point()) and dirichlet(v2.point()): 
            Facet[facet] = Dirichlet
        # Cells for Neumann condition
        if neumann(v1.point()) and neumann(v2.point()): 
            Facet[facet] = Neumann
    ds = df.Measure("ds", mesh, subdomain_data = Facet)

    # Variationnal problem
    a = df.inner(sigma(u), epsilon(v))*df.dx 
    L = df.dot(f,v)*df.dx + df.dot(g,v)*ds(Neumann)
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
'''
plt.figure()
plt.loglog(hh_phi,error_h1_phi,'o--', label=r'$\phi$-FEM $H^1$')
plt.loglog(hh_phi,error_l2_phi,'o-', label=r'$\phi$-FEM $L^2$')
plt.loglog(hh_standard,error_h1_standard, '--x',label=r'Std FEM $H^1$')
plt.loglog(hh_standard,error_l2_standard, '-x',label=r'Std FEM $L^2$')
if degV == 1 :
    plt.loglog(hh_phi, hh_phi, '.', label="Linear")
    plt.loglog(hh_phi,[hhh**2 for hhh in hh_phi], '.',label="Quadratic")
elif degV == 2 :
    plt.loglog(hh_phi,[hhh**2 for hhh in hh_phi], '.',label="Quadratic")
    plt.loglog(hh_phi,[hhh**3 for hhh in hh_phi], '.',label="Cubic")

plt.xlabel("$h$")
plt.ylabel(r'$\frac{\|u-u_h\|}{\|u\|}$')
plt.legend(loc='upper right', ncol=2)
plt.title(r'Relative error : $ \frac{\|u-u_h\|}{\|u\|} $ for $L^2$ and $H^1$ norms', y=1.025)
plt.tight_layout()
plt.savefig('relative_error_Phi_Fem_Mixed_Elasticity_P_{name0}.png'.format(name0=degV))
plt.show()

plt.figure()
plt.loglog(hh_phi,Time_assemble_phi, '-o',label=r'Assemble $\phi$-FEM')
plt.loglog(hh_phi,Time_solve_phi,'--o', label=r'Solve $\phi$-FEM')
plt.loglog(hh_standard,Time_assemble_standard, '-x',label=r'Assemble standard FEM')
plt.loglog(hh_standard,Time_solve_standard,'--x', label=r'Solve standard FEM')
plt.xlabel("$h$")
plt.ylabel("Time (s)")
plt.legend(loc='upper right')
plt.title("Computing time")
plt.tight_layout()
plt.savefig('Time_precision_Phi_Fem_Mixed_Elasticity_P_{name0}.png'.format(name0=degV))
plt.show()
plt.figure()
plt.loglog(error_l2_phi,Time_assemble_phi, '-o',label=r'Assemble $\phi$-fem')
plt.loglog(error_l2_phi,Time_solve_phi,'--o', label=r'Solve $\phi$-fem')
plt.loglog(error_l2_standard,Time_assemble_standard,'-x', label="Assemble standard FEM")
plt.loglog(error_l2_standard,Time_solve_standard,'--x', label="Solve standard FEM")
plt.xlabel(r'$\frac{\|u-u_h\|_{L^2}}{\|u\|_{L^2}}$')
plt.ylabel("Time (s)")
plt.title(r'Computing time')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Time_error_P_Phi_Fem_Mixed_Elasticity_{name0}.png'.format(name0=degV))
plt.show()
plt.figure()
plt.loglog(error_l2_phi,Time_total_phi,'-o', label=r'$\phi$-fem')
plt.loglog(error_l2_standard,Time_total_standard,'-x', label="Standard FEM")
plt.xlabel(r'$\frac{\|u-u_h\|_{L^2}}{\|u\|_{L^2}}$')
plt.ylabel("Time (s)")
plt.title(r'Computing time')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Total_time_error_Phi_Fem_Mixed_Elasticity_P_{name0}.png'.format(name0=degV))
plt.show()
'''

#  Write the output file for latex
f = open('output_Phi_Fem_Mixed_Elasticity_P{name0}.txt'.format(name0=degV),'w')
f.write('relative L2 norm phi fem: \n')	
output_latex(f, hh_phi, error_l2_phi)
f.write('relative H1 norm phi fem : \n')	
output_latex(f, hh_phi, error_h1_phi)
f.write('relative L2 norm and time phi fem : \n')	
output_latex(f, error_l2_phi, Time_total_phi)
f.write('relative H1 norm and time phi fem : \n')	
output_latex(f, error_h1_phi, Time_total_phi)
f.write('relative L2 norm classic fem: \n')	
output_latex(f, hh_standard, error_l2_standard)
f.write('relative H1 normclassic fem : \n')	
output_latex(f, hh_standard, error_h1_standard)
f.write('relative L2 norm and time classic fem : \n')	
output_latex(f, error_l2_standard, Time_total_standard)
f.write('relative H1 norm and time classic fem : \n')	
output_latex(f, error_h1_standard, Time_total_standard)
f.close()
