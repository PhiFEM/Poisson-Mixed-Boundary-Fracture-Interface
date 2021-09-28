import dolfin as df 
import matplotlib.pyplot as plt 
import multiphenics as mph 
import mshr
import numpy as np

# dolfin parameters
df.parameters["ghost_mode"] = "shared_facet" 
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters['allow_extrapolation'] = True
df.parameters["form_compiler"]["representation"] = 'uflacs'

# functions and parameters for elasticity
def sigma(u):
    return lambda_ * df.div(u)*df.Identity(2) + 2.0*mu*epsilon(u)

def epsilon(u):
    return (1.0/2.0)*(df.grad(u) + df.grad(u).T)


E = 0.07
nu = 0.3

lambda_ = E*nu/((1.0+nu)*(1.0-2.0*nu))
mu = E/(2.0*(1.0+nu))

# degree of interpolation for V and Vphi
degV = 2
degPhi = 2 + degV

# Function used to write in the outputs files
def output_latex(f,A,B):
	for i in range(len(A)):
		f.write('(')
		f.write(str(A[i]))
		f.write(',')
		f.write(str(B[i]))
		f.write(')\n')
	f.write('\n')
fi = open('output_holes_P{name0}.txt'.format(name0=degV),'w')

start_dist, end_dist, step_dist = 0,3,1 
for j in range(start_dist, end_dist, step_dist): # we increase the distance between the two centers of the circles       
    class phi_expr(df.UserExpression):
        def eval(self, value, x):
            # the value is here phi = - phi_1 * phi_2 (where phi_1 and phi_2 define the first and the second holes)
            value[0] = -(-0.15**2 + (x[0]-0.75)**2 + (x[1]-0.75)**2 )*(-(0.15)**2 + (x[0]-(0.50 -0.1*j))**2 + (x[1]-(0.50 -0.1*j))**2)
        def value_shape(self):
            return (2,)
    class phi1_expr(df.UserExpression):
        def eval(self, value, x):
            # the value is here phi = - phi_1 * phi_2 (where phi_1 and phi_2 define the first and the second holes)
            value[0] = -(-0.15**2 + (x[0]-0.75)**2 + (x[1]-0.75)**2 )
        def value_shape(self):
            return (2,)
    class phi2_expr(df.UserExpression):
        def eval(self, value, x):
            # the value is here phi = - phi_1 * phi_2 (where phi_1 and phi_2 define the first and the second holes)
            value[0] = -(-(0.15)**2 + (x[0]-(0.50 -0.1*j))**2 + (x[1]-(0.50 -0.1*j))**2)
        def value_shape(self):
            return (2,)        
    error_l2_phi, error_h1_phi, hh_phi = [], [], []
    start,end,step = 1,6,1
    for i in range(start,end,step): 
        print("Phi-fem iteration : ", i)
        H = 10*2**(i) 
        background_mesh = df.UnitSquareMesh(H,H)
        
        # we create Omega_h and Omega_h^Gamma phi
        V_phi = df.FunctionSpace(background_mesh, "CG", degPhi)
        phi = phi_expr(element = V_phi.ufl_element()) 
        phi = df.interpolate(phi, V_phi)

        phi1 = phi1_expr(element = V_phi.ufl_element()) 
        phi1 = df.interpolate(phi1, V_phi)
        
        phi2 = phi2_expr(element = V_phi.ufl_element()) 
        phi2 = df.interpolate(phi2, V_phi)
        Cell_omega = df.MeshFunction("size_t", background_mesh, background_mesh.topology().dim())
        Cell_omega.set_all(0)
        for cell in df.cells(background_mesh):  
            v1,v2,v3 = df.vertices(cell)
            if(phi1(v1.point()) < 0.0 or phi1(v2.point()) < 0.0 or phi1(v3.point()) < 0.0
               or df.near(phi1(v1.point()),0.0) or df.near(phi1(v2.point()),0.0) or df.near(phi1(v3.point()),0.0)):
                    if(phi2(v1.point()) < 0.0 or phi2(v2.point()) < 0.0 or phi2(v3.point()) < 0.0
                         or df.near(phi2(v1.point()),0.0) or df.near(phi2(v2.point()),0.0) or df.near(phi2(v3.point()),0.0)):
                            Cell_omega[cell] = 1
        mesh = df.SubMesh(background_mesh, Cell_omega, 1)   
        hh_phi.append(mesh.hmax()) # store the size of each element for this iteration  
        print('Background Mesh h =', hh_phi[-1])
        V_phi = df.FunctionSpace(mesh, "CG", degPhi)
        phi = phi_expr(element = V_phi.ufl_element()) 
        phi = df.interpolate(phi, V_phi)  
        phi1 = phi1_expr(element = V_phi.ufl_element()) 
        phi1 = df.interpolate(phi1, V_phi)
        
        phi2 = phi2_expr(element = V_phi.ufl_element()) 
        phi2 = df.interpolate(phi2, V_phi)
        # Selection of cells and facets on the boundary and creation of the restriction 
        hole = 4
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

        for cell in df.cells(mesh) :
            for facet in df.facets(cell): 
                v1,v2 = df.vertices(facet) 
                if(phi(v1.point())*phi(v2.point()) < 0.0 or df.near(phi(v1.point())*phi(v2.point()),0.0)) : 
                    Cell[cell] = hole 
                    cell_sub[cell] = 1
                    for facett in df.facets(cell):  
                        Facet[facett] = hole
                        facet_sub[facett] = 1
                        v1, v2 = df.vertices(facett)
                        vertices_sub[v1], vertices_sub[v2] = 1,1
                        
        File2 = df.File("Hole.rtc.xml/mesh_function_2.xml")
        File2 << cell_sub
        File1 = df.File("Hole.rtc.xml/mesh_function_1.xml")
        File1 << facet_sub
        File0 = df.File("Hole.rtc.xml/mesh_function_0.xml")
        File0 << vertices_sub
        Hole = mph.MeshRestriction(mesh,"Hole.rtc.xml")  
        
        # creation of the spaces
        V = df.VectorFunctionSpace(mesh,"CG",degV, dim=2)
        Z = df.TensorFunctionSpace(mesh,"CG",degV, shape = (2,2))
        Q = df.VectorFunctionSpace(mesh,"DG",degV-1, dim = 2)
        W = mph.BlockFunctionSpace([V,Z,Q], restrict=[None,Hole,Hole])
        uyp = mph.BlockTrialFunction(W)
        (u, y,p) = mph.block_split(uyp)
        vzq = mph.BlockTestFunction(W)
        (v, z,q) = mph.block_split(vzq)
        
        # modification of the measures to integrate on Omega_h^Gamma
        dx = df.Measure("dx", mesh, subdomain_data = Cell)
        ds = df.Measure("ds", mesh, subdomain_data = Facet)
        dS = df.Measure("dS", mesh, subdomain_data = Facet)

        # parameters, constants and expressions        
        gamma_div, gamma_u, gamma_p, sigma_p = 1.0, 1.0, 1.0, 0.01
        h = df.CellDiameter(mesh)
        n = df.FacetNormal(mesh)
        u_D = df.Constant((0.0,0.0)) # Dirichlet condition (standard conditions on the square)
        def boundary(x, on_boundary):
            return on_boundary and (df.near(x[1],0.0))
        bc1 = mph.DirichletBC(W.sub(0), u_D, boundary) # apply DirichletBC on Omega
        bcs = mph.BlockDirichletBC([bc1])
        f = df.Constant(('0.0','-0.9'))

        # Construction of the bilinear and linear forms
        boundary_penalty = sigma_p*df.avg(h)*df.inner(df.jump(sigma(u),n), df.jump(sigma(v),n))*dS(hole) \
        
        auv = df.inner(sigma(u), epsilon(v))*dx \
            + gamma_u*df.inner(sigma(u),sigma(v))*dx(hole) \
            + boundary_penalty 
        auz = gamma_u*df.inner(sigma(u),z)*dx(hole) 
        auq = 0.0
        
        ayv = df.inner(df.dot(y,n),v)*ds(hole) + gamma_u*df.inner(y,sigma(v))*dx(hole) 
        ayz = gamma_u*df.inner(y,z)*dx(hole) + gamma_div*df.inner(df.div(y), df.div(z))*dx(hole) \
            + gamma_p*h**(-2)*df.inner(df.dot(y,df.grad(phi)), df.dot(z,df.grad(phi)))*dx(hole)         
        ayq = gamma_p*h**(-3)*df.inner(df.dot(y,df.grad(phi)), q*phi)*dx(hole) 
        
        apv = 0.0
        apz = gamma_p*h**(-3)*df.inner(p*phi, df.dot(z,df.grad(phi)))*dx(hole) 
        apq = gamma_p*h**(-4)*df.inner(p*phi,q*phi)*dx(hole)

        lv = df.inner(f,v)*dx  
        lz = gamma_div*df.inner(f, df.div(z))*dx(hole) 
        lq = 0.0
        
        # construction of the FE matrices and resolution 
        a = [[auv,auz,auq],
            [ayv,ayz,ayq],
            [apv,apz,apq]]
        l = [lv,lz,lq]
        A = mph.block_assemble(a)
        B = mph.block_assemble(l)
        bcs.apply(A)
        bcs.apply(B)
        UU = mph.BlockFunction(W)
        mph.block_solve(A, UU.block_vector(), B, linear_solver="mumps") 
        u_h_phi = UU[0]
        if i == end - 1 : # distance between the two holes
            c1 = np.array([0.75,0.75])
            c2 = np.array([0.50-0.1*j,0.50-0.1*j])
            dist = np.linalg.norm(c2-c1)
        
        # Compute standard FEM (fine mesh)
        if i == start :
            domain = mshr.Rectangle(df.Point(0,0), df.Point(1,1)) \
                - mshr.Circle(df.Point(0.50-0.1*j,0.50-0.1*j), 0.15) - mshr.Circle(df.Point(0.75,0.75), 0.15)
            H = 320
            mesh = mshr.generate_mesh(domain,H)
            print("Standard fem iteration : ", i)
            print("Fine mesh h =", mesh.hmax())
            # FunctionSpace Pk
            V_std = df.VectorFunctionSpace(mesh, 'CG', degV, dim=2)
            u, v = df.TrialFunction(V_std), df.TestFunction(V_std)
            u_D = df.Constant((0.0,0.0))
            f = df.Expression(('0.0','-0.9'), degree=1, domain = mesh)
            # Resolution of the variationnal problem
            def boundary(x, on_boundary):
                return on_boundary and (df.near(x[1],0.0))
            a = df.inner(sigma(u), epsilon(v))*df.dx
            l = df.inner(f,v)*df.dx 
            
            bc = df.DirichletBC(V_std, u_D, boundary) # apply DirichletBC on Omega
            start_standard = time.time()
            u = df.Function(V_std)
            df.solve(a==l, u, bc)
            u_h_std = u
            u_h_std = df.project(u_h_std, V_std)
            end_standard = time.time()
            
        u_h_phi = df.project(u_h_phi, V_std) # project the solution of the phi-fem on the fine mesh

        # compute and store L2 and H1 errors between phi-FEM and Standard FEM
        error_l2_phi.append((df.assemble((((u_h_std-u_h_phi))**2)*df.dx)**(0.5))/(df.assemble((((u_h_std))**2)*df.dx)**(0.5)))            
        error_h1_phi.append((df.assemble(((df.grad(u_h_std- u_h_phi))**2)*df.dx)**(0.5))/(df.assemble(((df.grad(u_h_std))**2)*df.dx)**(0.5)))
        print(error_l2_phi[-1])
    print('Dist = {:.6f}'.format(dist - 2* 0.15) + ' L2 error = ', error_l2_phi)
    fi.write('Fine mesh h ='+ str(mesh.hmax()) + ' \n')
    fi.write('Dist = {:.6f} '.format(dist -2* 0.15) + ' relative L2 norm phi fem: \n')	
    output_latex(fi, hh_phi, error_l2_phi)
    fi.write('relative H1 norm phi fem : \n')	
    output_latex(fi, hh_phi, error_h1_phi)
fi.close()

