import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xji, self.yij = ...

        linspace = np.linspace(0,1,N+1)
        self.xij, self.yij = np.meshgrid(linspace,linspace, indexing='ij')
        

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1,-2,1],[-1,0,1],(N+1,N+1), 'lil')
        D[0,:4] = 2, -5, 4, -1
        D[-1,-4:] = -1, 4, -5, 2
        D = (N**2)*D
        return D
        

    @property
    def w(self):
        """Return the dispersion coefficient"""
        #find from putting exact sol into PDE adn getin a relationship between w aand c, ky,kx
        return self.c*np.pi*np.sqrt(self.mx**2+self.my**2)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.create_mesh(N)
        self.dx = 1/N
        c, dt, D = self.c, self.dt, self.D2(N)
        self.Unp1, self.Un, self.Unm1 = np.zeros((3, N+1, N+1))
        self.Unm1[:] = sp.lambdify((x,y),self.ue(mx,my).subs({t: 0}))(self.xij, self.yij)
        self.Un[:] = sp.lambdify((x,y),self.ue(mx,my).subs({t: self.dt}))(self.xij, self.yij)# this was in lecture 7: self.Unm1[:] + 0.5*(c*dt)**2*(D @ self.Un + self.Un @ D.T)
        self.data = {0: self.Unm1.copy()}
        

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.dx/self.c
        

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        e = sp.lambdify((x,y),self.ue(self.mx,self.my).subs({t:t0}))(self.xij,self.yij) - u 
        l2error = np.sqrt(np.sum(e**2)  *  self.dx**2)
        return l2error

    def apply_bcs(self):
        Unp1 = self.Unp1
        Unp1[0] = 0
        Unp1[-1] = 0
        Unp1[:, -1] = 0
        Unp1[:, 0] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.c = c
        self.cfl = cfl
        self.dx = 1/N
        self.mx, self.my = mx, my
        self.initialize(N, mx, my)
        
        
        
        
        dt = self.dt
        D = self.D2(N)
        Unp1, Un, Unm1 = self.Unp1, self.Un, self.Unm1 
        if store_data == 1:
            self.data[1] = Un.copy()
        for n in range(1, Nt):
            Unp1[:] = 2*Un - Unm1 + (c*dt)**2*(D @ Un + Un @ D.T)
            # Set boundary conditions
            self.apply_bcs()
            # Swap solutions
            Unm1[:] = Un
            Un[:] = Unp1
            if n % store_data == 0:
                self.data[n] = Unm1.copy() # Unm1 is now swapped to Un
        if store_data >0:
            return self.data
        if store_data == -1:
            return self.dx, self.l2_error(Unm1.copy(), dt*(Nt-1))
        

    def convergence_rates(self, m=7, cfl=0.1, Nt=10, mx=3, my=3): #increased m from 4 to 7 to pass test
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)  #changed from err[-1] so that we only need compute l2error for the last time step
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        #print(r) for testing
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = sparse.diags([1,-2,1],[-1,0,1],(N+1,N+1), 'lil')
        D[0,1] =2
        D[-1,-2] = 2
        D = (N**2)*D
        return D

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        #we implement them in the D matrix
        pass

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    sol = Wave2D()
    solN = Wave2D_Neumann()
    #checks for mx = my and cfl = 1/sqrt(2)
    for i in range(2,20):
        N0 , Nt , cfl , mx , my = 100, 100, 1/np.sqrt(2), i, i
        dx, err = sol(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
        dxN, errN = solN(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
        #print(err,errN) 
        assert abs(err) < 1e-12
        assert abs(errN) < 1e-12

