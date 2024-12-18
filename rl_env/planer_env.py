
import numpy as np
class PlanerEnv:
    def __init__(self):
        self.dt = 0.1
        self.step_count = 0
    def reset(self):
        s  = np.sign( np.random.randn(2) )
        r  = np.random.rand(2)
        X1  = s[0]*r[0]*1.0
        X2  = s[1]*r[1]*1.0  #sign(r(2))*0.8 + sign(r(3))*0.35*rand;
        T   = 2.0
        self.state = np.array([X1, X2,T])
        return self.state

    def step(self, action):       
        action = np.clip(action,-1,1)
        # Current state
        X1 = self.state[0]
        X2 = self.state[1]
        T  = self.state[2]
        rho = 1
        # New state
        U  = action
        
        new_state = np.zeros_like(self.state)
        new_state[0]  =  X1 + self.dt*( - X1**3- X2)
        new_state[1]  =  X2 + self.dt*( X1   + X2 + rho*U )
        new_state[2]  =  T -self.dt
        
        # Check terminal conditios 
        issafe     = X1**2 + X2**2 < 0.1
        isUnsafe   = abs( X1 ) > 2 or abs(X2) > 2
        done       = issafe or isUnsafe
        
        #Reward 
        StageCost  = -(X1*X2 + U ** 2) / 10
        
        if not done :
            reward = StageCost
        elif issafe:
             reward = 0
        else :
             reward = -100
                
        self.state = new_state
        self.step_count += 1
        
        if self.step_count >= 100:
            done = True
        
        return new_state, reward, done
        
