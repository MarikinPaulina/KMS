import numpy as np
from tqdm.autonotebook import tqdm
from pathlib import Path
from numba import njit

@njit
def H(phi, t, N, X, kappa, omega):
    pot =  kappa * (X - 0.5)* phi * np.sin(omega*t)
    kin = -0.5 * N**2 * (np.roll(phi,1) + np.roll(phi,-1) - 2*phi)
    H = kin+pot
    H[0] = 0
    H[-1] = 0
    return H

class Simulation:
    
    def __init__(self, n, N, kappa, omega, dt, s_data, s_pos, file_name, steps):
        self.n = n
        self.N = N
        self.kappa = kappa
        self.omega = omega
        self.dt = dt
        self.s_data = s_data
        self.s_pos = s_pos
        self.file_name = file_name
        self.steps = steps
        
        self.X = np.linspace(0,1,self.N+1)
        self.phi_re = 2**0.5 * np.sin(n*np.pi*self.X)
        self.phi_im = np.zeros_like(self.phi_re)
          
    def step(self, i):
        self.phi_re += H(self.phi_im, i*self.dt, self.N, self.X, self.kappa, self.omega) * self.dt * 0.5
        self.phi_im -= H(self.phi_re, i*self.dt, self.N, self.X, self.kappa, self.omega) * self.dt
        self.phi_re += H(self.phi_im, i*self.dt, self.N, self.X, self.kappa, self.omega) * self.dt * 0.5
        
        if i%self.s_data == 0:
            self.save_data(i)
        if i%self.s_pos == 0:
            self.save_pos(i)
        
    def save_data(self, i):
        N_ = (self.phi_re**2 + self.phi_im**2).sum() / self.N
        x = (self.X * (self.phi_re**2 + self.phi_im**2)).sum() / self.N
        eps = (self.phi_re*H(self.phi_re, i*self.dt, self.N, self.X, self.kappa, self.omega) + 
               self.phi_im*H(self.phi_im, i*self.dt, self.N, self.X, self.kappa, self.omega)).sum() / self.N
        with open(f'{self.file_name}.out','a') as _file:
            _file.write(f'{N_},{x},{eps}\n')
            
    def save_pos(self, i):
        ro = self.phi_re[::2]**2 + self.phi_im[::2]**2
        with open(f'{self.file_name}.dat', 'ab') as _file:
            np.savetxt(_file,[ro],delimiter=',',fmt='%.6e')
        
    def simulation(self):
        self.X = np.linspace(0,1,self.N+1)
        self.phi_re = 2**0.5 * np.sin(self.n*np.pi*self.X)
        self.phi_im = np.zeros_like(self.phi_re)
        
        Path(self.file_name).mkdir(parents=True, exist_ok=True)
        with open(f'{self.file_name}.out','w') as _file:
            _file.write('N,x,E\n')
        open(f'{self.file_name}.dat', 'w')
        
        for i in tqdm(range(self.steps)):
            self.step(i)
            
                
        Path(self.file_name).rmdir()
