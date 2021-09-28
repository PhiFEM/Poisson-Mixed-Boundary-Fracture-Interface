import dolfin as df 
import matplotlib.pyplot as plt 
import mshr
from vedo.dolfin import plot, interactive
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




# select the mesh to plot
# 1 : dirichlet standard
# 2 : dirichlet phifem
# 3 : mixed standard conform
# 4 : mixed phifem conform
# 5 : mixed standard non-conform
# 6 : mixed phifem non-coform
# 7 : interface standard
# 8 : interface phi fem
# 9 : fracture 
select = 5




class phi_expr(df.UserExpression) : 
    def eval(self, value, x):
        value[0] = -1.0/8.0 + (x[0]-0.5)**2 + (x[1]-0.5)**2 
    def value_shape(self):
        return (1,)

# expression of phi
class phi_expr_inter(df.UserExpression) : 
    def eval(self, value, x):
        value[0] =  -R**2 + (x[0]-0.5)**2 + (x[1] - 0.5)**2 
    def value_shape(self):
        return (1,)

# expression of phi
class phi_expr_fracture(df.UserExpression) : 
    def eval(self, value, x):
        value[0] = x[1] - 0.25*df.sin(2*df.pi*x[0]) - .5
    def value_shape(self):
        return (1,)

if select == 1 or select == 3 or select == 5:
    # Computation of the standard FEM       
    domain = mshr.Circle(df.Point(0.5,0.5),df.sqrt(2.0)/4.0) # creation of the domain
    if select == 3:
        domain.set_subdomain(1, mshr.Rectangle(df.Point(0.0, 0.0), df.Point(0.5, 1.0))) 
    mesh = mshr.generate_mesh(domain,7)
    mesh.rotate(20.0,2,df.Point(0.5,0.5))
    plot(mesh,c="white")
    if select == 3 or select == 5:
        domain = mshr.Circle(df.Point(0.5,0.5),df.sqrt(2.0)/4.0+0.005) -mshr.Circle(df.Point(0.5,0.5),df.sqrt(2.0)/4.0-0.005)
        mesh2 = mshr.generate_mesh(domain,50)
        Cell = df.MeshFunction("size_t", mesh2, mesh2.topology().dim())
        Cell.set_all(0)
        for cell in df.cells(mesh2):  
            if (cell.midpoint().x()>=0.5 and select == 3) or (cell.midpoint().x()>=0.5 and select == 5):
                Cell[cell] = 1
            if (cell.midpoint().x()<=0.5 and select == 3) or (cell.midpoint().x()<=0.5 and select == 5):
                Cell[cell] = 2
        mesh_dir = df.SubMesh(mesh2, Cell, 1) 
        mesh_neu = df.SubMesh(mesh2, Cell, 2) 
        plot(mesh_dir,c="red",lw=0.0,add=True)
        plot(mesh_neu,c="blue",lw=0.0,add=True)


if select == 2 or select == 4 or select == 6:
    # Computation of the standard FEM 
    if select ==  4:   
        background_mesh = df.UnitSquareMesh(10,10)  
    else:
        background_mesh = df.UnitSquareMesh(9,9)

    # Creation of Omega_h
    V_phi = df.FunctionSpace(background_mesh, "CG", 1)
    phi = phi_expr(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    Cell_omega = df.MeshFunction("size_t", background_mesh, background_mesh.topology().dim())
    Cell_omega.set_all(0)
    for cell in df.cells(background_mesh):  
        v1,v2,v3 = df.vertices(cell)
        if(phi(v1.point()) <= 0.0 or phi(v2.point()) <= 0.0 or phi(v3.point()) <= 0.0 or df.near(phi(v1.point()),0.0) or df.near(phi(v2.point()),0.0) or df.near(phi(v3.point()),0.0)):
            Cell_omega[cell] = 1
    mesh = df.SubMesh(background_mesh, Cell_omega, 1) 
    plot(mesh,c="white",add=True)

    if select ==2:
        # selection of Omega_h^Gamma
        V_phi = df.FunctionSpace(mesh, "CG", 1)
        phi = phi_expr(element = V_phi.ufl_element())
        phi = df.interpolate(phi, V_phi)
        Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())
        Cell.set_all(0)
        for cell in df.cells(mesh):  
            for facet in df.facets(cell):
                v1,v2 = df.vertices(facet)
                if phi(v1.point())*phi(v2.point()) <= 0.0 or df.near(phi(v1.point())*phi(v2.point()),0.0):
                    Cell[cell] = 1
        mesh = df.SubMesh(mesh, Cell, 1) 
        plot(mesh,c="yellow",add=True)

    if select ==4 or select == 6:
        # selection of Omega_h^Gamma
        V_phi = df.FunctionSpace(mesh, "CG", 1)
        phi = phi_expr(element = V_phi.ufl_element())
        phi = df.interpolate(phi, V_phi)
        Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())
        Cell.set_all(0)
        for cell in df.cells(mesh):  
            for facet in df.facets(cell):
                v1,v2 = df.vertices(facet)
                if phi(v1.point())*phi(v2.point()) <= 0.0 or df.near(phi(v1.point())*phi(v2.point()),0.0):
                    Cell[cell] = 1
                    v1,v2,v3=df.vertices(cell)
                    if v1.point().x()>=0.5 and v2.point().x()>=0.5 and v3.point().x()>=0.5:
                        Cell[cell] = 2
                    if v1.point().x()<=0.5 and v2.point().x()<=0.5 and v3.point().x()<=0.5:
                        Cell[cell] = 3
        mesh_dirichlet = df.SubMesh(mesh, Cell, 2) 
        mesh_neumann = df.SubMesh(mesh, Cell, 3) 
        mesh_rest = df.SubMesh(mesh, Cell, 1) 
        plot(mesh_dirichlet,c="red",add=True)
        plot(mesh_neumann,c="blue",add=True)
        plot(mesh_rest,c="yellow",add=True)


    domain = mshr.Circle(df.Point(0.5,0.5),df.sqrt(2.0)/4.0+0.001) -mshr.Circle(df.Point(0.5,0.5),df.sqrt(2.0)/4.0-0.001)
    mesh2 = mshr.generate_mesh(domain,50)
    plot(mesh2,c="black",add=True)


if select == 7:
    R= 0.3
    domain = mshr.Rectangle(df.Point(0.0, 0.0), df.Point(1.0, 1.0)) # creation of the domain
    domain.set_subdomain(1, mshr.Circle(df.Point(0.5,0.5),R)) 
    H = 10 # to have approximately the same precision as in the phi-fem computation
    mesh = mshr.generate_mesh(domain,H)
    plot(mesh,c="white",add=True)
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    Cell.set_all(0)
    for cell in df.cells(mesh):  
        v = cell.midpoint()
        if -R**2 + (v.x()-0.5)**2 + (v.y() - 0.5)**2 <0:
            Cell[cell] = 1
    mesh = df.SubMesh(mesh, Cell, 1) 
    plot(mesh,add=True)


if select == 8:
    R= 0.3
    H = 8
    mesh = df.UnitSquareMesh(H,H)
    plot(mesh,c="white",add=True)
    V_phi = df.FunctionSpace(mesh, "CG", 4)
    phi = phi_expr_inter(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())  
    Cell.set_all(0)    
    # creation of Omega1 (including the interface)
    for cell in df.cells(mesh) :
        for facet in df.facets(cell):
            v1,v2 = df.vertices(facet)
            if phi(v1.point())*phi(v2.point()) < 0.0  or df.near(phi(v1.point())*phi(v2.point()),0.0):
                Cell[cell] = 1 
    mesh = df.SubMesh(mesh, Cell, 1) 
    plot(mesh,add=True)
    domain = mshr.Circle(df.Point(0.5,0.5),R+0.001) -mshr.Circle(df.Point(0.5,0.5),R-0.001)
    mesh2 = mshr.generate_mesh(domain,50)
    plot(mesh2,c="black",add=True)


if select == 9:
    H = 16
    # creation of the domain (check which cells are on the fracture)
    mesh = df.RectangleMesh(df.Point(.0,.0), df.Point(1,1),H,H)
    plot(mesh,c="white",add=True)

    V_phi = df.FunctionSpace(mesh, "CG", 4)
    phi = phi_expr_fracture(element = V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    
    # initialization of mesh functions to create Omega1, Omega2 and the boundaries
    mesh.init(1,2)  
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())  
    Cell.set_all(0)
    for cell in df.cells(mesh):  
        for facet in df.facets(cell): 
            v1,v2 = df.vertices(facet) 
            if((phi(v1.point())*phi(v2.point()) < 0.0 or df.near(phi(v1.point())*phi(v2.point()), 0.0))):
                Cell[cell] = 3
                vc1, vc2, vc3 = df.vertices(cell)
                if (vc1.point().x() <= 0.5 ) and (vc2.point().x() <= 0.5) and (vc3.point().x() <= 0.5):
                    Cell[cell] = 1
                if (vc1.point().x() >= 0.5 ) and (vc2.point().x() >= 0.5) and (vc3.point().x() >= 0.5):
                    Cell[cell] = 2
    mesh1 = df.SubMesh(mesh, Cell, 1) 
    plot(mesh1,c="blue",add=True)
    mesh2 = df.SubMesh(mesh, Cell, 2) 
    plot(mesh2,c='red',add=True)
    mesh3 = df.SubMesh(mesh, Cell, 3) 
    plot(mesh3,c='yellow',add=True)



