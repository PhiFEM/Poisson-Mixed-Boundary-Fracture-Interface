import dolfin as df 
import matplotlib.pyplot as plt 
from matplotlib import rc, rcParams
import multiphenics as mph 

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

# Function used to write in the outputs files
def output_latex(f,A,B):
	for i in range(len(A)):
		f.write('(')
		f.write(str(A[i]))
		f.write(',')
		f.write(str(B[i]))
		f.write(')\n')
	f.write('\n')
 
# degree of interpolation for V and Vphi
degV = 2
degPhi = 2 + degV 


def sigma(u):
    return Lambda * df.div(u)*df.Identity(2) + 2.0*mu*epsilon(u)

def epsilon(u):
    return (1.0/2.0)*(df.grad(u) + df.grad(u).T)


E = 7
nu = 0.3

Lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))
mu = E/(2.0*(1.0+nu))


# expression of phi
class phi_expr(df.UserExpression) : 
    def eval(self, value, x):
        value[0] = x[1] - 0.25*df.sin(2*df.pi*x[0]) - .5
    def value_shape(self):
        return (2,)


error_l2=[]
error_h1=[]
hh = []

start,end,step= 1,6,1
for i in range(start,end,step):
    print('###########################')
    print('### iteration ',i,'###')
    print('###########################')
    H = 10*2**i+1
    # creation of the domain (check which cells are on the fracture)
    mesh = df.RectangleMesh(df.Point(.0,.0), df.Point(1,1),H,H)
    hh.append(mesh.hmax())

    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    
    # initialization of mesh functions to create Omega1, Omega2 and the boundaries
    omega1, omega2, interf, gamma1, gamma2, fracture, gamma1_N, gamma2_N, fracture_interf, gamma_fracture_interf1, gamma_fracture_interf2 = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11
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


    # yellow cells
    for cell in df.cells(mesh):  
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if((phi(v1.point())*phi(v2.point()) < 0.0 or df.near(phi(v1.point())*phi(v2.point()), 0.0))):
                Cell[cell] = fracture_interf
                for facett in df.facets(cell):  
                    Facet[facett] = fracture_interf


    # creation of the restricition for the interface
    cell_sub.set_all(0)
    facet_sub.set_all(0)
    vertices_sub.set_all(0)
    for cell in df.cells(mesh):  
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if((phi(v1.point())*phi(v2.point()) < 0.0 or df.near(phi(v1.point())*phi(v2.point()), 0.0))):
                vc1, vc2, vc3 = df.vertices(cell)
                if (vc1.point().x() <= 0.5 ) and (vc2.point().x() <= 0.5) and (vc3.point().x() <= 0.5):
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
    
    # Creation of the boundaries of the two domains                     
    for cell in df.cells(mesh):
        if Cell[cell] == omega2 :
            for facet in df.facets(cell) :
                if Facet[facet] == interf :
                    Facet[facet] = gamma1

        if Cell[cell] == omega1 :
            for facet in df.facets(cell) :
                if Facet[facet] == interf :
                    Facet[facet] = gamma2 
                    
    # Creation of the restriction for the fracture           
    cell_sub.set_all(0)
    facet_sub.set_all(0)
    vertices_sub.set_all(0)
    for cell in df.cells(mesh):  
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if((phi(v1.point())*phi(v2.point()) < 0.0 or df.near(phi(v1.point())*phi(v2.point()), 0.0))):
                vc1, vc2, vc3 = df.vertices(cell)
                if (vc1.point().x() >= 0.5 ) and (vc2.point().x() >= 0.5) and (vc3.point().x() >= 0.5):
                    Cell[cell] = fracture
                    cell_sub[cell] = 1
                    for facett in df.facets(cell):  
                        Facet[facett] = fracture
                        facet_sub[facett] = 1
                        v1, v2 = df.vertices(facett)
                        vertices_sub[v1], vertices_sub[v2] = 1,1
                        
    File2 = df.File("Fracture.rtc.xml/mesh_function_2.xml")
    File2 << cell_sub
    File1 = df.File("Fracture.rtc.xml/mesh_function_1.xml")
    File1 << facet_sub
    File0 = df.File("Fracture.rtc.xml/mesh_function_0.xml")
    File0 << vertices_sub
    Fracture = mph.MeshRestriction(mesh,"Fracture.rtc.xml")


    # Determine which facets are on the which boundary 
    for cell in df.cells(mesh):
        if Cell[cell] == omega2 :
            for facet in df.facets(cell) :
                if Facet[facet] == fracture :
                    Facet[facet] = gamma1_N

        if Cell[cell] == omega1 :
            for facet in df.facets(cell) :
                if Facet[facet] == fracture :
                    Facet[facet] = gamma2_N 
    
    for cell in df.cells(mesh):
        if Cell[cell] == fracture_interf :
            print('toto')
        if Cell[cell] == omega1:
            for facet in df.facets(cell):
                if Facet[facet] == fracture_interf :
                    Facet[facet] = gamma_fracture_interf2     
        if Cell[cell] == omega2:
            for facet in df.facets(cell):
                if Facet[facet] == fracture_interf :
                    Facet[facet] = gamma_fracture_interf1  
    
    # creation of the spaces 
    V = df.VectorFunctionSpace(mesh,'CG',degV, dim=2)
    Z = df.TensorFunctionSpace(mesh,"CG",degV, shape = (2,2))
    Q = df.VectorFunctionSpace(mesh,"DG",degV-1, dim = 2)
    W = mph.BlockFunctionSpace([V,V,Z,Z,Q,Z,Q,Z,Q], 
                        restrict=[Omega1, Omega2, Interface, Interface, Interface,
                                Fracture, Fracture, Fracture, Fracture])
    uyp = mph.BlockTrialFunction(W)
    (u1, u2, y1, y2, p, y1_N, p1_N, y2_N, p2_N) = mph.block_split(uyp)
    vzq = mph.BlockTestFunction(W)
    (v1, v2, z1, z2, q, z1_N, q1_N, z2_N, q2_N) = mph.block_split(vzq)
    dx = df.Measure("dx", mesh, subdomain_data = Cell)
    ds = df.Measure("ds", mesh, subdomain_data = Facet)
    dS = df.Measure("dS", mesh, subdomain_data = Facet)


    h = df.CellDiameter(mesh)
    # normal to the mesh on Gamma
    n = df.FacetNormal(mesh)
    
    # parameters for the test case
    u_ex = df.Expression(('sin(x[0])*exp(x[1])', 'cos(x[1])*exp(x[0])'), domain = mesh, degree = 4)

    f = -df.div(sigma(u_ex))
    u_D = u_ex
    u_D = df.project(u_D,V)
    phi_abs = df.inner(df.grad(phi),df.grad(phi))**0.5


    g = df.dot(sigma(u_ex),df.grad(phi))/phi_abs + u_ex*phi 
    
    gamma_div, gamma_u, gamma_p, gamma_y = 1.0, 1.0, 1.0, 1.0
    gamma_div_N, gamma_u_N, gamma_p_N = 1.0, 1.0, 1.0
    sigma_jump = 1.0

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
    Gh1 = sigma_jump*df.avg(h)*df.inner(df.jump(sigma(u1),n), df.jump(sigma(v1),n))*(dS(gamma2)+dS(gamma2_N)+dS(gamma_fracture_interf2))     
    Gh2 = sigma_jump*df.avg(h)*df.inner(df.jump(sigma(u2),n), df.jump(sigma(v2),n))*(dS(gamma1)+dS(gamma1_N)+dS(gamma_fracture_interf1))

    
    au1v1 = df.inner(sigma(u1), epsilon(v1))*(dx(omega1) + dx(interf) + dx(fracture) + dx(fracture_interf))  \
        + Gh1 \
        + gamma_p*h**(-2)*df.inner(u1,v1)*dx(interf) \
        + gamma_u*df.inner(sigma(u1), sigma(v1))*dx(interf) \
        + gamma_u_N*df.inner(sigma(u1), sigma(v1))*dx(fracture) - (w("+")*df.inner(df.dot(sigma(u1)("+"),n("+")),v1("+"))+w("-")*df.inner(df.dot(sigma(u1)("-"),n("-")),v1("-")))*dS(gamma_fracture_interf1) #\
        #+ sigma_jump*h**2*df.inner(df.div(sigma(u1)),df.div(sigma(v1)))*dx(fracture_interf)

    au1z1 = gamma_u*df.inner(sigma(u1), z1)*dx(interf)
    au1v2 = -gamma_p*h**(-2)*df.inner(u1,v2)*dx(interf)
    au1q = gamma_p*h**(-3)*df.inner(u1,q*phi)*dx(interf)
    au1z1_N = gamma_u_N*df.inner(sigma(u1),z1_N)*dx(fracture) 

    
    au2v1 = -gamma_p*h**(-2)*df.inner(u2,v1)*dx(interf)
    au2v2 = df.inner(sigma(u2), epsilon(v2))*(dx(omega2) + dx(interf) + dx(fracture) + dx(fracture_interf)) \
        + Gh2 \
        + gamma_p*h**(-2)*df.inner(u2,v2)*dx(interf) \
        + gamma_u*df.inner(sigma(u2), sigma(v2))*dx(interf) \
        + gamma_u_N*df.inner(sigma(u2), sigma(v2))*dx(fracture)- (w("+")*df.inner(df.dot(sigma(u2)("+"),n("+")),v2("+"))+w("-")*df.inner(df.dot(sigma(u2)("-"),n("-")),v2("-")))*dS(gamma_fracture_interf2) #\
        #+ sigma_jump*h**2*df.inner(df.div(sigma(u2)),df.div(sigma(v2)))*dx(fracture_interf)
    au2z2 = gamma_u*df.inner(sigma(u2), z2)*dx(interf)
    au2q = -gamma_p*h**(-3)*df.inner(u2,q*phi)*dx(interf)
    au2z2_N = gamma_u_N*df.inner(sigma(u2),z2_N)*dx(fracture) 
    
    ay1v1 = (w("+")*df.inner(df.dot(y1("+"),n("+")),v1("+"))+w("-")*df.inner(df.dot(y1("-"),n("-")),v1("-")))*dS(gamma1) \
            + gamma_u*df.inner(y1,sigma(v1))*dx(interf)
    ay1z1 = gamma_div*df.inner(df.div(y1),df.div(z1))*dx(interf)\
            + gamma_u*df.inner(y1,z1)*dx(interf) \
            + gamma_y*h**(-2)*df.inner(df.dot(y1,df.grad(phi)),df.dot(z1,df.grad(phi)))*dx(interf)
    ay1z2 = - gamma_y*h**(-2)*df.inner(df.dot(y1,df.grad(phi)),df.dot(z2,df.grad(phi)))*dx(interf)
    
    
    ay2v2 = (w("+")*df.inner(df.dot(y2("+"),n("+")),v2("+"))+w("-")*df.inner(df.dot(y2("-"),n("-")),v2("-")))*dS(gamma2) \
            + gamma_u*df.inner(y2,sigma(v2))*(dx(interf))
    ay2z1 = -gamma_y*h**(-2)*df.inner(df.dot(y2,df.grad(phi)),df.dot(z1,df.grad(phi)))*dx(interf)  
    ay2z2 = gamma_div*df.inner(df.div(y2),df.div(z2))*(dx(interf))\
            + gamma_u*df.inner(y2,z2)*(dx(interf)) \
            + gamma_y*h**(-2)*df.inner(df.dot(y2,df.grad(phi)),df.dot(z2,df.grad(phi)))*(dx(interf))


    apv1 =  gamma_p*h**(-3)*df.inner(p*phi,v1)*dx(interf)
    apv2 = -gamma_p*h**(-3)*df.inner(p*phi,v2)*dx(interf)
    apq  = gamma_p*h**(-4)*df.inner(p*phi,q*phi)*(dx(interf))

    
    ay1_Nv1 = (w("+")*df.inner(df.dot(y1_N("+"),n("+")),v1("+"))+w("-")*df.inner(df.dot(y1_N("-"),n("-")),v1("-")))*dS(gamma1_N) \
            + gamma_u_N*df.inner(y1_N,sigma(v1))*dx(fracture) 
    ay1_Nz1_N = gamma_u_N*df.inner(y1_N,z1_N)*dx(fracture) \
                + gamma_div_N*df.inner(df.div(y1_N), df.div(z1_N))*dx(fracture) \
                + gamma_p_N*h**(-2)*df.inner(df.dot(y1_N,df.grad(phi)), df.dot(z1_N,df.grad(phi)))*dx(fracture)
    ay1_Nq1_N = gamma_p_N*h**(-3)*df.inner(df.dot(y1_N,df.grad(phi)), q1_N*phi)*dx(fracture)

    ap1_Nz1_N = gamma_p_N*h**(-3)*df.inner(p1_N*phi, df.dot(z1_N,df.grad(phi)))*dx(fracture)
    ap1_Nq1_N = gamma_p_N*h**(-4)*df.inner(p1_N*phi,q1_N*phi)*dx(fracture)   

    ay2_Nv2 = (w("+")*df.inner(df.dot(y2_N("+"),n("+")),v2("+"))+w("-")*df.inner(df.dot(y2_N("-"),n("-")),v2("-")))*dS(gamma2_N) \
            + gamma_u_N*df.inner(y2_N,sigma(v2))*dx(fracture)
    ay2_Nz2_N = gamma_u_N*df.inner(y2_N,z2_N)*dx(fracture) \
                + gamma_div_N*df.inner(df.div(y2_N), df.div(z2_N))*dx(fracture) \
                + gamma_p_N*h**(-2)*df.inner(df.dot(y2_N,df.grad(phi)), df.dot(z2_N,df.grad(phi)))*dx(fracture)
    ay2_Nq2_N = gamma_p_N*h**(-3)*df.inner(df.dot(y2_N,df.grad(phi)), q2_N*phi)*dx(fracture)
        
    ap2_Nz2_N = gamma_p_N*h**(-3)*df.inner(p2_N*phi, df.dot(z2_N,df.grad(phi)))*dx(fracture)
    ap2_Nq2_N = gamma_p_N*h**(-4)*df.inner(p2_N*phi, q2_N*phi)*dx(fracture)         
    
            
    lv1 = df.dot(f,v1)*(dx(omega1) + dx(interf) + dx(fracture) + dx(fracture_interf)) #\
         #- sigma_jump*h**2*df.inner(f,df.div(sigma(v1)))*dx(fracture_interf)
    lv2 = df.dot(f,v2)*(dx(omega2) + dx(interf) + dx(fracture) + dx(fracture_interf)) # \
         #- sigma_jump*h**2*df.inner(f,df.div(sigma(v2)))*dx(fracture_interf)
    lz1 = gamma_div * df.inner(f,df.div(z1))*(dx(interf)) 
    lz2 = gamma_div * df.inner(f,df.div(z2))*(dx(interf)) 
    lz1_N = gamma_div_N * df.inner(f,df.div(z1_N))*(dx(fracture)) \
            - gamma_p_N*h**(-2)*df.inner(g*phi_abs, df.dot(z1_N,df.grad(phi)))*dx(fracture)
    lq1_N = - gamma_p_N*h**(-3)*df.inner(g*phi_abs,q1_N*phi)*dx(fracture)
    lz2_N = gamma_div_N * df.inner(f,df.div(z2_N))*(dx(fracture)) \
            - gamma_p_N*h**(-2)*df.inner(g*phi_abs, df.dot(z2_N,df.grad(phi)))*dx(fracture)
    lq2_N = - gamma_p_N*h**(-3)*df.inner(g*phi_abs,q2_N*phi)*dx(fracture)
    
    
    A = [[au1v1, au1v2, au1z1, 0.0, au1q, au1z1_N, 0.0, 0.0, 0.0], 
            [au2v1, au2v2, 0.0, au2z2, au2q, 0.0, 0.0, au2z2_N, 0.0],
            [ay1v1, 0.0, ay1z1, ay1z2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, ay2v2, ay2z1, ay2z2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [apv1,  apv2, 0.0, 0.0,  apq,  0.0,  0.0,  0.0,  0.0], 
            [ay1_Nv1, 0.0, 0.0, 0.0, 0.0, ay1_Nz1_N, ay1_Nq1_N, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, ap1_Nz1_N, ap1_Nq1_N, 0.0, 0.0],
            [0.0, ay2_Nv2, 0.0, 0.0, 0.0, 0.0, 0.0, ay2_Nz2_N, ay2_Nq2_N],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ap2_Nz2_N, ap2_Nq2_N]]
    
    L = [lv1, lv2, lz1, lz2, 0.0, lz1_N, lq1_N, lz2_N, lq2_N]

    print('bilinear and linear forms : ok')

    AA = mph.block_assemble(A)
    LL = mph.block_assemble(L)

    print('assemble : ok')

    # definition of the Dirichlet boundary (top, bottom, left and right sides of the mesh)
    def boundary(x, on_boundary):
        return on_boundary and (df.near(x[0], 0.0) or df.near(x[0], 1.0) or df.near(x[1],1.0) or df.near(x[1],0.0))
    bc1 = mph.DirichletBC(W.sub(0), u_D, boundary) # apply DirichletBC on Omega1
    bc2 = mph.DirichletBC(W.sub(1), u_D, boundary) # apply DirichletBC on Omega2
    bcs = mph.BlockDirichletBC([bc1,bc2])
    bcs.apply(AA)
    bcs.apply(LL)

    print('boundary conditions : ok')

    UU = mph.BlockFunction(W)
    mph.block_solve(AA, UU.block_vector(), LL)
    # Solution on Omega1
    u_h1 = df.project(UU[0], V)
    # Solution on Omega2
    u_h2 = df.project(UU[1], V)
    # Solution on Omega 
    #u_h = b*UU[0] + (1.0-b)*UU[1]
    #u_h = df.project(u_h, V)
    u_ex = df.project(u_ex, V)
    
    error_l2.append(df.assemble((u_ex-UU[0])**2*dx(omega1)+(u_ex-UU[1])**2*(dx(omega2)+dx(fracture)+dx(interf)))**(0.5)/df.assemble(u_ex**2*df.dx)**(0.5))            
    error_h1.append((df.assemble((df.grad(u_ex-UU[0])**2)*dx(omega1)+df.grad(u_ex-UU[1])**2*(dx(omega2)+df.dx(fracture)+dx(interf)))**(0.5))/(df.assemble(df.grad(u_ex)**2*df.dx)**(0.5)))          
    print('h = ',mesh.hmax())
    print('erreur L2 = ',error_l2[-1])
    print('erreur H1 = ',error_h1[-1])
    
#  Write the output file for latex
f = open('outputs/output_fracture_P{name0}.txt'.format(name0=degV),'w')
f.write('relative L2 norm phi fem: \n')	
output_latex(f, hh, error_l2)
f.write('relative H1 norm phi fem : \n')	
output_latex(f, hh, error_h1)
f.close()
