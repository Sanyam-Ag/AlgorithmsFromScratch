import numpy as np

class SNE:
    def __init__(self, perplexity = 10, lr = 0.1, seed = None, trackloss = False):
        self.perplexity = perplexity
        self.lr = lr
        self.rng = np.random.default_rng(seed=seed)  # generator

        # original data distribution metrics
        self.P_ij = None
        self.distp = None
        self.sigma = None

        # testing prints
        self.test = False

        # tracking loss
        self.trackloss = trackloss
        self.lossArr = []


    def __call__(self):
        print("SNE: use fit() method for low dimensionality output")

        
    def fit(self, X:np.ndarray, newdims = 2, iters=10):
        """
            X_dims[0] = no of data points
            x_dims[1] = no of dimensions in high dimensional space
        """
  
        X_dims = X.shape
        Y = self.rng.standard_normal((X_dims[0], newdims))   #generator for Y

        # Original datapoint probability distribution
        self.P_ij = self._calcPij(X, X_dims)

        distq=self._calcdist(Y)
        Q_ij = self._calcQij(Y, distq)

        # Loop for training low dimensional sample space
        for i in range(iters):

            grad = self._calcgrad(Y, Q_ij)
            Y = Y - self.lr * grad

            distq=self._calcdist(Y)
            Q_ij = self._calcQij(Y, distq)

            if self.test:
                print("Q: ", Q_ij, "\n")
                print(np.min(Q_ij))
                print(np.max(Q_ij))
                print(np.sum(Q_ij, axis=1))

            if self.trackloss:  # trackloss is used as calculating loss is wasted computation though its helpful in seeing optimization of lower dim sample space
                Loss = self._calcentropy(self.P_ij, Q_ij)
                print("Loss", i, ":", Loss)
                self.lossArr.append(Loss)

        return Y



    # Calculating Euclidean distance
    def _calcdist(self, dpoints: np.ndarray) -> np.ndarray:
        """ 
            shape = dpoints.shape
            npoints = shape[0]
            ndims = shape[1]
            distmat = np.zeros((npoints, npoints))
            for i in range(npoints):
                distmat[i][i] = 0
                for j in range(i+1, npoints):
                    ans = 0
                    for k in range(ndims):
                        ans += ((dpoints[i][k] - dpoints[j][k])**2)
                    distmat[i][j]  = distmat[j][i] = math.sqrt(ans)
            return distmat 
        """
        
        x = np.asarray(dpoints, dtype=np.float64)
        sq_norms = np.sum(x * x, axis=1)
        dist2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (x @ x.T)

        # Protect against tiny negative values caused by floating-point error.
        np.maximum(dist2, 0, out=dist2)

        return np.sqrt(dist2)

        
    # Calculating the asymmetric probability for original data distributino
    def _calcPij(self, X:np.ndarray, X_dims):
        self.distp = self._calcdist(X)
        
        # sigma is used only in the original distribution
        self.sigma = self._calcgaussianbandwidth()
        n = X_dims[0]
        P = np.zeros((n, n))

        """for i in range(n):
            for j in range(n):
                if i == j:  continue
                P[i, j] = np.exp(-(self.distp[i, j] ** 2) / (2 * self.sigma[i] ** 2))

            # Normalize p(j|i)
            P[i] /= np.sum(P[i])  """

        for i in range(n):
            distances = np.delete(self.distp[i], i)

            logits = -(distances ** 2) / (2 * self.sigma[i] ** 2)
            logits -= np.max(logits)

            probs = np.exp(logits)
            probs /= np.sum(probs)

            mask = np.arange(n) != i
            P[i, mask] = probs

        return P


    # Gaussian Bandwidth calculation used in Pij to match the set perplexity
    def _calcgaussianbandwidth(self):   #sigma
        n = self.distp.shape[0]
        sigmas = np.zeros(n)

        target_entropy = np.log(self.perplexity)                        

        for i in range(n):
            low = 1e-10                      
            high = 1.0

            # Distances from point i to all other points
            distances = np.delete(self.distp[i], i)
            # Find an upper bound for sigma - adaptive searching
            while True:                       

                p = np.exp(-(distances ** 2) / (2 * high ** 2))
                p /= np.sum(p)

                entropy = self._calcentropy(p, p)
                if entropy >= target_entropy: break                    

                high *= 2.0

            # Binary search for sigma
            for _ in range(50):
                sigma = (low + high) / 2.0
                logits = -(distances ** 2) / (2 * sigma ** 2)
                logits -= np.max(logits)    # prevents p from becoming all zeroes
                p = np.exp(logits)
                p /= np.sum(p)

                entropy = self._calcentropy(p, p)
                if entropy < target_entropy:    low = sigma              
                else:   high = sigma             

            sigmas[i] = (low + high) / 2.0

        return sigmas


    #Calculating the low dimensional space asymmetric porbability
    def _calcQij(self, Y:np.ndarray, distq):
        n = Y.shape[0]
        Q = np.zeros((n, n))

        for i in range(n):
            distances = np.delete(distq[i], i)

            logits = -(distances ** 2)
            logits -= np.max(logits)

            probs = np.exp(logits)
            probs /= np.sum(probs)

            mask = np.arange(n) != i
            Q[i, mask] = probs

        return Q


    # Calculating gradient: del(Loss)/ del(Yi)
    def _calcgrad(self, Y, Q):
        n = Y.shape[0]
        diff = Y[:, None, :] - Y[None, :, :]
        weights = self.P_ij - Q + self.P_ij.T - Q.T
        np.fill_diagonal(weights, 0.0)
        
        return 2.0 * np.sum(weights[:, :, None] * diff, axis=1)


    # Effective cost function as Pij is constant so KL divergence loss is effectively cross entropy loss
    def _calcentropy(self, D1:np.ndarray, D2:np.ndarray):     # D1, D2 are probability distributions
        D2 = np.clip(D2, 1e-12, 1.0)
        return -np.sum(D1 * np.log(D2))
