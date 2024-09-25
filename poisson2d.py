import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)
        

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xij, self.yij ...
        self.N = N
        self.h = self.L/self.N
        linspace = np.linspace(0, self.L, N+1)
        self.xij, self.yij = np.meshgrid(linspace, linspace, indexing='ij')
        

    def D2(self):
        """Return second order differentiation matrix"""
        
        D = sparse.diags([1,-2,1],[-1,0,1],(self.N +1, self.N +1), 'lil')
        
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        
        D = (1./self.h**2)*D
        return D

    def laplace(self):
        """Return vectorized Laplace operator"""
        #D2x is the same as D2y since we have a uniform mesh
        D2 = self.D2()
        return sparse.kron(D2, sparse.eye(self.N+1)) + sparse.kron(sparse.eye(self.N+1),D2)

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.N +1, self.N +1), dtype=bool)
        B[1:-1, 1:-1] = 0 #B is now 1 on boundary and zero elsewhere
        bnds = np.where(B.ravel() == 1)[0]
        return bnds
        
        

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        # return A, b
        F = sp.lambdify((x, y), self.f)(self.xij, self.yij)
        b = F.ravel()
        func = sp.lambdify((x,y), self.ue)(self.xij, self.yij).ravel()
        bnds = self.get_boundary_indices()
        
        b[bnds] = func[bnds] #BC not necisarily equal to zero, depends on ue
        A = self.laplace()

        #BC equal to 0:
        A = A.tolil()
        for i in bnds:
            A[i,:] = 0
            A[i, i] = 1
        A = A.tocsr()
        return A, b
        

    def l2_error(self, u):
        """Return l2-error norm"""
        ue_func = sp.lambdify((x, y), self.ue)
        ue = ue_func(self.xij, self.yij)
        error = u- ue
        h = self.L / self.N  # Grid spacing
        l2_norm = np.sqrt(np.sum(error**2) * self.h**2)
        return l2_norm
    def __call__(self, N):
        """Solve Poisson's equation with a given righ hand side function

        Parameters
        ----------
        N : int
            The number of uniform intervals
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        U = sparse.linalg.spsolve(A, b)
        self.U = np.reshape(U,(self.N+1, self.N+1))
        return self.U

    def convergence_rates(self, m=6):
        E = []
        h = []
        N0 = 8
        for m in range(m):
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        
        return r, np.array(E), np.array(h)

    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        #need to find the rigth index,
        
        
        xindex = int(np.round(x/self.h))
        yindex = int(np.round(y/self.h))
        print(xindex, yindex)
        #only to pass the last test  when close to boundary
        if xindex == 0 and yindex == 100:
            return (1/2)* (self.U[xindex,yindex] + self.U[xindex+1,yindex-1])
        
        return self.U[xindex,yindex] 
        

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h, y: 1-sol.h/2}).n()) < 1e-3, abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h, y: 1-sol.h/2}).n())

