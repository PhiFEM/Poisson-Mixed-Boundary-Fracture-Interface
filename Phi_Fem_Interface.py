#docker pull multiphenics/multiphenics
#sudo docker run --shm-size=15g --cpus=8 -ti -v $(pwd):/home/fenics/shared multiphenics/multiphenics
import dolfin as df 
import matplotlib.pyplot as plt 
import mshr
from matplotlib import rc, rcParams
import multiphenics as mph
import time

# plot parameters
plt.style.use('bmh') 
params = {'axes.labelsize': 'large',
          'font.size': 22,
          'axes.titlesize': 'large',
          'legend.fontsize': 18,
          'figure.titlesize': 24,
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
degPhi = 1 + degV 


# elastic parameters
E1 = 7.0
nu1 = 0.3
lambda_1 = E1*nu1/((1.0+nu1)*(1.0-2.0*nu1))
mu1 = E1/(2.0*(1.0+nu1))
E2 = 2.28
nu2 = 0.3
lambda_2 = E2*nu2/((1.0+nu2)*(1.0-2.0*nu2))
mu2 = E2/(2.0*(1.0+nu2))

# size of the circle in the domain
R = 0.3

# expression of phi
class phi_expr(df.UserExpression) : 
    def eval(self, value, x):
        value[0] =  -R**2 + (x[0]-0.5)**2 + (x[1] - 0.5)**2 

    def value_shape(self):
        return (1,)

# functions and parameters for elasticity
def sigma1(u):
    return lambda_1 * df.div(u)*df.Identity(2) + 2.0*mu1*epsilon(u)

def sigma2(u):
    return lambda_2 * df.div(u)*df.Identity(2) + 2.0*mu2*epsilon(u)

def epsilon(u):
    return (1.0/2.0)*(df.grad(u) + df.grad(u).T)



class sol_exact(df.UserExpression):
    def eval(self, value, x):
        r = pow(pow(x[0]-0.5,2) + pow(x[1] - 0.5,2),0.5)
        a = df.as_vector((df.cos(r)-df.cos(R), df.cos(r)-df.cos(R) ))
        if r<R:
            u = a/E1
        else:
            u = a/E2
        value[0] = u[0]
        value[1] = u[1]
    def value_shape(self):
        return (2,) 

def f_exact():
    aa = df.Expression(('cos(pow(pow(x[0]-0.5,2) + pow(x[1] - 0.5,2),0.5))','cos(pow(pow(x[0]-0.5,2) + pow(x[1] - 0.5,2),0.5))'),degree=8, domain = mesh)
    return -df.div(sigma1(aa))/E1



time_phi_fem, error_h1_phi_fem, error_l2_phi_fem, hh_phi_fem = [], [],[],[]
start,end,step= 0,1,1
for i in range(start,end,step):
    print("iteration : ", i)
    H = 10*2**i
    # creation of the domain
    mesh = df.UnitSquareMesh(H,H)
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    hh_phi_fem.append(mesh.hmax())
    
    # initialization of mesh functions to create Omega1, Omega2 and the boundaries
    omega1, omega2, interf, gamma1, gamma2 = 1, 2, 3, 4, 5
    mesh.init(1,2) 
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1) 
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())  
    cell_sub = df.MeshFunction("bool", mesh, mesh.topology().dim())
    facet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim()-1)
    vertices_sub = df.MeshFunction("bool", mesh, 0)
    Facet.set_all(0)
    Cell.set_all(0)
    cell_sub.set_all(0)
    facet_sub.set_all(0)
    vertices_sub.set_all(0)
    
    # creation of Omega1 (including the interface)
    for cell in df.cells(mesh) :
        v1,v2,v3 = df.vertices(cell)
        if(phi(v1.point()) < 0.0 or phi(v2.point()) < 0.0 or phi(v3.point()) < 0.0 
        or df.near(phi(v1.point()),0.0) or df.near(phi(v2.point()),0.0) or df.near(phi(v3.point()),0.0)):
            Cell[cell] = omega1 
            cell_sub[cell] = 1
            for facett in df.facets(cell):  
                Facet[facett] = omega1
                facet_sub[facett] = 1
                v1, v2 = df.vertices(facett)
                vertices_sub[v1], vertices_sub[v2] = 1,1

    File2 = df.File("omega1.rtc.xml/mesh_function_2.xml")
    File2 << cell_sub
    File1 = df.File("omega1.rtc.xml/mesh_function_1.xml")
    File1 << facet_sub
    File0 = df.File("omega1.rtc.xml/mesh_function_0.xml")
    File0 << vertices_sub
    Omega1 = mph.MeshRestriction(mesh,"omega1.rtc.xml")
    
    # Creation of Omega2 (including the interface)
    cell_sub.set_all(0)
    facet_sub.set_all(0)
    vertices_sub.set_all(0)
    for cell in df.cells(mesh) :
        v1,v2,v3 = df.vertices(cell)
        if(phi(v1.point()) > 0.0 or phi(v2.point()) > 0.0 or phi(v3.point()) > 0.0 
            or df.near(phi(v1.point()),0.0) or df.near(phi(v2.point()),0.0) or df.near(phi(v3.point()),0.0)):
            Cell[cell] = omega2 
            cell_sub[cell] = 1
            for facett in df.facets(cell):
                Facet[facett] = omega2  
                facet_sub[facett] = 1
                v1, v2 = df.vertices(facett)
                vertices_sub[v1], vertices_sub[v2] = 1,1
    File2 = df.File("omega2.rtc.xml/mesh_function_2.xml")
    File2 << cell_sub
    File1 = df.File("omega2.rtc.xml/mesh_function_1.xml")
    File1 << facet_sub
    File0 = df.File("omega2.rtc.xml/mesh_function_0.xml")
    File0 << vertices_sub
    Omega2 = mph.MeshRestriction(mesh,"omega2.rtc.xml")

    # creation of the restricition for the interface
    cell_sub.set_all(0)
    facet_sub.set_all(0)
    vertices_sub.set_all(0)
    for cell in df.cells(mesh):  
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if(phi(v1.point())*phi(v2.point()) < 0.0 or df.near(phi(v1.point())*phi(v2.point()), 0.0)): 
                Cell[cell] = interf
                cell_sub[cell] = 1
                for facett in df.facets(cell):  
                    Facet[facett] = interf
                    facet_sub[facett] = 1
                    v1, v2 = df.vertices(facett)
                    vertices_sub[v1], vertices_sub[v2] = 1,1
                
    File2 = df.File("interface.rtc.xml/mesh_function_2.xml")
    File2 << cell_sub
    File1 = df.File("interface.rtc.xml/mesh_function_1.xml")
    File1 << facet_sub
    File0 = df.File("interface.rtc.xml/mesh_function_0.xml")
    File0 << vertices_sub
    Interface = mph.MeshRestriction(mesh,"interface.rtc.xml")
    
    for cell in df.cells(mesh):
        if Cell[cell] == omega2 :
            for facet in df.facets(cell) :
                if Facet[facet] == interf :
                    Facet[facet] = gamma1

        if Cell[cell] == omega1 :
            for facet in df.facets(cell) :
                if Facet[facet] == interf :
                    Facet[facet] = gamma2 
            
    # creation of the spaces 
    V = df.VectorFunctionSpace(mesh, 'CG', degV, dim=2)
    Z = df.TensorFunctionSpace(mesh,"CG",degV, shape = (2,2))
    if degV == 1:	
        Q = df.VectorFunctionSpace(mesh,"DG",degV, dim = 2)
    else:
        Q = df.VectorFunctionSpace(mesh,"CG",degV, dim = 2)
    W = mph.BlockFunctionSpace([V,V,Z,Z,Q], restrict=[Omega1, Omega2, Interface, Interface, Interface])
    uyp = mph.BlockTrialFunction(W)
    (u1, u2, y1, y2, p) = mph.block_split(uyp)
    vzq = mph.BlockTestFunction(W)
    (v1, v2, z1, z2, q) = mph.block_split(vzq)
    
    # modification of the measures
    dx = df.Measure("dx", mesh, subdomain_data = Cell)
    ds = df.Measure("ds", mesh, subdomain_data = Facet)
    dS = df.Measure("dS", mesh, subdomain_data = Facet)
    
    # parameters for the considered case
    gamma_div, gamma_u, gamma_p, gamma_y, sigma_p = 10.0, 10.0, 10.0, 10.0, 0.1
    h = df.CellDiameter(mesh)
    n = df.FacetNormal(mesh)    

    V_ex = df.VectorFunctionSpace(mesh, 'CG', degV+2, dim=2)
    u_ex = sol_exact(element = V_ex.ufl_element())
    f = f_exact()

    u_ex = df.interpolate(u_ex, V_ex)
    u_D = u_ex 
    
    # DG for the interface
    DG0 = df.FunctionSpace(mesh,'DG',0)
    w = df.Function(DG0)
    for c in range(mesh.num_cells()):
        mycell = df.Cell(mesh,c)
        for facet in df.facets(mycell): 
            vert1,vert2 = df.vertices(facet) 
            if(phi(vert1.point())*phi(vert2.point()) < 0.0 or df.near(phi(vert1.point())*phi(vert2.point()), 0.0)): 
                w.vector()[c] = 1.0

    # Construction of the bilinear and linear forms
    start_assemble = time.time()
    Gh1 = sigma_p*df.avg(h)*df.inner(df.jump(sigma1(u1),n), df.jump(sigma1(v1),n))*(dS(interf)+dS(gamma2)) 
    Gh2 = sigma_p*df.avg(h)*df.inner(df.jump(sigma2(u2),n), df.jump(sigma2(v2),n))*(dS(interf)+dS(gamma1))
    
    au1v1 = df.inner(sigma1(u1), epsilon(v1))*(dx(omega1) + dx(interf))  + Gh1 \
        + gamma_p*h**(-2)*df.inner(u1,v1)*dx(interf) \
        + gamma_u*df.inner(sigma1(u1), sigma1(v1))*dx(interf) 
    au1z1 = gamma_u*df.inner(sigma1(u1), z1)*dx(interf)
    au1v2 = -gamma_p*h**(-2)*df.inner(u1,v2)*dx(interf)
    au1q = gamma_p*h**(-3)*df.inner(u1,q*phi)*dx(interf)
    
    au2v1 = -gamma_p*h**(-2)*df.inner(u2,v1)*dx(interf)
    au2v2 = df.inner(sigma2(u2), epsilon(v2))*(dx(omega2) + dx(interf)) + Gh2 \
        + gamma_p*h**(-2)*df.inner(u2,v2)*dx(interf) \
        + gamma_u*df.inner(sigma2(u2), sigma2(v2))*dx(interf)  
    au2z2 = gamma_u*df.inner(sigma2(u2), z2)*dx(interf)
    au2q = -gamma_p*h**(-3)*df.inner(u2,q*phi)*dx(interf)
    
    ay1v1 = (w("+")*df.inner(df.dot(y1("+"),n("+")),v1("+"))+w("-")*df.inner(df.dot(y1("-"),n("-")),v1("-")))*dS(gamma1) + gamma_u*df.inner(y1,sigma1(v1))*dx(interf)
    ay1z1 = gamma_div*df.inner(df.div(y1),df.div(z1))*dx(interf) + gamma_u*df.inner(y1,z1)*dx(interf) \
            + gamma_y*h**(-2)*df.inner(df.dot(y1,df.grad(phi)),df.dot(z1,df.grad(phi)))*dx(interf)
    ay1z2 = - gamma_y*h**(-2)*df.inner(df.dot(y1,df.grad(phi)),df.dot(z2,df.grad(phi)))*dx(interf)

    ay2v2 = (w("+")*df.inner(df.dot(y2("+"),n("+")),v2("+"))+w("-")*df.inner(df.dot(y2("-"),n("-")),v2("-")))*dS(gamma2) + gamma_u*df.inner(y2,sigma2(v2))*dx(interf)
    ay2z1 = -gamma_y*h**(-2)*df.inner(df.dot(y2,df.grad(phi)),df.dot(z1,df.grad(phi)))*dx(interf)  
    ay2z2 = gamma_div*df.inner(df.div(y2),df.div(z2))*dx(interf) \
            + gamma_u*df.inner(y2,z2)*dx(interf) \
            + gamma_y*h**(-2)*df.inner(df.dot(y2,df.grad(phi)),df.dot(z2,df.grad(phi)))*dx(interf)

    apv1 =  gamma_p*h**(-3)*df.inner(p*phi,v1)*dx(interf)
    apv2 = -gamma_p*h**(-3)*df.inner(p*phi,v2)*dx(interf)
    apq  = gamma_p*h**(-4)*df.inner(p*phi,q*phi)*dx(interf)
    
    lv1 = df.dot(f,v1)*(dx(omega1) + dx(interf)) 
    lv2 = df.dot(f,v2)*(dx(omega2) + dx(interf)) 
    lz1 = gamma_div * df.inner(f,df.div(z1))*dx(interf) 
    lz2 = gamma_div * df.inner(f,df.div(z2))*dx(interf) 

    A = [[au1v1, au1v2, au1z1, 0.0, au1q], 
            [au2v1, au2v2, 0.0, au2z2, au2q],
            [ay1v1, 0.0, ay1z1, ay1z2, 0.0],
            [0.0, ay2v2, ay2z1, ay2z2, 0.0],
            [apv1,  apv2, 0.0, 0.0,  apq]]
    L = [lv1, lv2, lz1, lz2, 0.0]

    AA = mph.block_assemble(A)
    LL = mph.block_assemble(L)
    # definition of the Dirichlet conditions (top, bottom, left and right sides of the square)
    def boundary(x, on_boundary):
        return on_boundary and (df.near(x[0], 0.0) or df.near(x[0], 1.0) or df.near(x[1],1.0) or df.near(x[1],0.0))
    bc2 = mph.DirichletBC(W.sub(1), u_D, boundary) # apply DirichletBC on Omega2
    bcs = mph.BlockDirichletBC([bc2])
    bcs.apply(AA)
    bcs.apply(LL)
    UU = mph.BlockFunction(W)
    mph.block_solve(AA, UU.block_vector(), LL)
    end_solve = time.time()

    # Solution on Omega1
    u_h1 = df.project(UU[0], V)
    # Solution on Omega2
    u_h2 = df.project(UU[1], V)    


    # Compute and store relative error for H1 and L2 norms
    relative_error_L2_phi_fem = df.assemble(df.inner(u_ex-u_h1,u_ex-u_h1)*dx(1)+df.inner(u_ex-u_h2,u_ex-u_h2)*dx(2))/df.assemble(df.inner(u_ex,u_ex)*dx(1)+df.inner(u_ex,u_ex)*dx(2))
    relative_error_L2_phi_fem = df.sqrt(relative_error_L2_phi_fem)
    print("Relative error L2 phi FEM : ",relative_error_L2_phi_fem)
    error_l2_phi_fem.append(relative_error_L2_phi_fem) 
    relative_error_H1_phi_fem = df.assemble(df.inner(df.grad(u_ex-u_h1),df.grad(u_ex-u_h1))*dx(1)+df.inner(df.grad(u_ex-u_h2),df.grad(u_ex-u_h2))*dx(2))/df.assemble(df.inner(df.grad(u_ex),df.grad(u_ex))*dx(1) + df.inner(df.grad(u_ex),df.grad(u_ex))*dx(2)) 
    relative_error_H1_phi_fem  = df.sqrt(relative_error_H1_phi_fem)
    error_h1_phi_fem.append(relative_error_H1_phi_fem) 
    print("Relative error H1 phi FEM : ",relative_error_H1_phi_fem)
    time_phi_fem.append(end_solve-start_assemble)
    print("time standard FEM : ",end_solve-start_assemble)


# Computation of the standard FEM       
domain = mshr.Rectangle(df.Point(0.0, 0.0), df.Point(1.0, 1.0)) # creation of the domain
domain.set_subdomain(1, mshr.Circle(df.Point(0.5,0.5),R)) 
time_standard, error_l2_standard, error_h1_standard, hh_standard = [], [],[],[]
for i in range(start, end, step):
    H = 8*2**(i) # to have approximately the same precision as in the phi-fem computation
    mesh = mshr.generate_mesh(domain,H)
    print("Standard fem iteration : ", i)
    # FunctionSpace P1
    V = df.VectorFunctionSpace(mesh, 'CG', degV, dim=2)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    n = df.FacetNormal(mesh)
    boundary = 'on_boundary '

    V_ex = df.VectorFunctionSpace(mesh, 'CG', degV+2, dim=2)
    u_ex = sol_exact(element = V_ex.ufl_element())
    f = f_exact()
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())

    # initialization of mesh functions to create Omega1, Omega2 and the boundaries
    omega1, omega2 = 1, 2
    mesh.init(1,2) 
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())  
    Cell.set_all(0)
    
    # creation of Omega1 (including the interface)
    for cell in df.cells(mesh) :
        if(phi(cell.midpoint())) < 0.0:
            Cell[cell] = omega1 
        else:
            Cell[cell] = omega2 
    dx = df.Measure("dx", mesh, subdomain_data = Cell)


    
    # Boundary conditions
    u_D = u_ex 
    bc = df.DirichletBC(V, u_D, boundary)

    # Variationnal problem
    a = df.inner(sigma1(u), epsilon(v))*dx(omega1) +df.inner(sigma2(u), epsilon(v))*dx(omega2) 
    L = df.dot(f,v)*df.dx 
    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(L)
    end_assemble = time.time()
    u = df.Function(V)
    bc.apply(A,B) # apply Dirichlet boundary conditions to the problem    
    start_solve = time.time()
    df.solve(A, u.vector(), B)
    end_solve = time.time()
    u = df.project(u, V_ex)
    u_ex = df.project(u_ex, V_ex)
    dx_ex = df.Measure("dx", mesh)

    # Compute and store h and L2 H1 errors
    hh_standard.append(mesh.hmax())
    relative_error_L2_standard_fem = df.sqrt(df.assemble((df.inner(u_ex-u,u_ex-u)*dx_ex)))/df.sqrt(df.assemble((df.inner(u_ex,u_ex))*dx_ex))
    error_l2_standard.append(relative_error_L2_standard_fem)  
    print("Relative error L2 standard FEM : ",relative_error_L2_standard_fem)
    relative_error_H1_standard_fem = df.sqrt(df.assemble((df.inner(df.grad(u_ex-u),df.grad(u_ex-u))*dx_ex)))/df.sqrt(df.assemble((df.inner(df.grad(u_ex),df.grad(u_ex)))*dx_ex))
    error_h1_standard.append(relative_error_H1_standard_fem) 
    print("Relative error H1 standard FEM : ",relative_error_H1_standard_fem)
    time_standard.append(end_solve-start_assemble)
    print("time standard FEM : ",end_solve-start_assemble)





"""# plot error on the figure matplotlib   
plt.loglog(hh_phi_fem,error_h1_phi_fem,'o--', label=r'$\phi$-FEM $H^1$')
plt.loglog(hh_phi_fem,error_l2_phi_fem,'o-', label=r'$\phi$-FEM $L^2$')
plt.xlabel("$h$")
plt.ylabel(r'$\frac{\|u-u_h\|}{\|u\|}$')
plt.legend(loc='lower right', ncol=2)
plt.title(r'Relative error : $ \frac{\|u-u_h\|}{\|u\|} $ for $L^2$ and $H^1$ norms', y=1.025)
plt.tight_layout()
plt.show()"""

# Function used to write in the outputs files
def output_latex(f,A,B):
	for i in range(len(A)):
		f.write('(')
		f.write(str(A[i]))
		f.write(',')
		f.write(str(B[i]))
		f.write(')\n')
	f.write('\n')

#  Write the output file for latex
f = open('elasticity_interface_P{name0}.txt'.format(name0=degV),'w')
f.write('(E_1, nu_1, lambda_1, mu_1) = ( ' + str(E1) + ', ' + str(nu1) + ', ' + str(lambda_1) + ', ' + str(mu1) + ') \n')  	
f.write('(E_2, nu_2, lambda_2, mu_2) = ( ' + str(E2) + ', ' + str(nu2) + ', ' + str(lambda_2) + ', ' + str(mu2) + ') \n')
f.write('relative L2 norm phi fem: \n')	
output_latex(f, hh_phi_fem, error_l2_phi_fem)
f.write('relative H1 norm phi fem : \n')	
output_latex(f, hh_phi_fem, error_h1_phi_fem)
f.write('time standard : \n')	
output_latex(f, error_l2_standard, time_standard)
f.write('relative L2 norm standard fem: \n')	
output_latex(f, hh_standard, error_l2_standard)
f.write('relative H1 norm standard fem : \n')	
output_latex(f, hh_standard, error_h1_standard)
f.write('time phi fem : \n')	
output_latex(f, error_l2_phi_fem, time_phi_fem)
f.close()
