import numpy as np

# K-nearest neighbor implementation
class KNN:
    def __init__(self, k=5, p=2):
        # passing p='inf' represents infinity
        self.k = k
        self.p = p
        
        pass
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        pass
    
    def predict(self, X_test):
        if self.X_train is None or self.y_train is None:
            return None
        
        y_test = []
        
        for x in X_test:
            neighbors = []
            
            for pos, x2 in enumerate(self.X_train):
                distance = 0
                if self.p == 'inf':
                    inf_dist = 0
                    
                    for feature1, feature2 in zip(x, x2):
                        if abs(feature1 - feature2) > inf_dist:
                            inf_dist = abs(feature1 - feature2)
                    
                    distance = inf_dist
                else:
                    for feature1, feature2 in zip(x, x2):
                        distance += abs(feature1 - feature2) ** self.p
                        
                    distance = distance ** (1/self.p)
                
                if len(neighbors) < self.k:
                    neighbors.append([distance, self.y_train[pos]])
                else:
                    max_dist = 0
                    del_pos = 0
                    
                    for pos2, neighbor in enumerate(neighbors):
                        if neighbor[0] > max_dist:
                            max_dist = neighbor[0]
                            del_pos = pos2
                    
                    if distance < max_dist:
                        del neighbors[del_pos]
                        neighbors.append([distance, self.y_train[pos]])
            
            classes = [neighbor[1] for neighbor in neighbors]
            y_test.append(max(set(classes), key=classes.count))
            
        return np.array(y_test)
        
        pass
    