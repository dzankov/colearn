from torch.utils.data import Dataset


class GetMB(Dataset):
    
    def __init__(self, X, y):
        
        super(GetMB, self).__init__()
        
        self.X = X
        self.y = y
    
    def __getitem__(self, i):
        
        return self.X[i], self.y[i]
    
    def __len__(self):
        
        return len(self.y)


		
class GetTauMB(Dataset):
    
    def __init__(self, X1, X2, y):
        
        super(GetTauMB, self).__init__()
        
        self.X1 = X1
        self.X2 = X2
        self.y = y
    
    def __getitem__(self, i):
        
        return self.X1[i], self.X2[i], self.y[i]
    
    def __len__(self):
        
        return len(self.y)


class GetArrheniusMB(Dataset):
    
    def __init__(self, X, y, T):
        
        super(GetArrheniusMB, self).__init__()
        
        self.X = X
        self.y = y
        self.T = T
    
    def __getitem__(self, i):
        
        return self.X[i], self.y[i], self.T[i]
    
    def __len__(self):
        
        return len(self.y)		
