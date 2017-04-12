import numpy as np

class ReplayCache():
    def __init__(self,cache_size,batch_size,image_size,mmap_fname=None,reuse=False):
        """
        optionally save cache to memory mapped file
        batch size is the size of the batch to get from the cache
        (this might be half (or other) of the batch size of the training)
        """
        self.cache_size = cache_size
        self.batch_size = int(batch_size)
        self.image_size = image_size
        self.nelements = 0
        self.elements = np.zeros([self.cache_size] + list(self.image_size))

        if mmap_fname is not None:
            if reuse:
                self.nelements=self.cache_size
            else:
                np.save(mmap_fname,self.elements)
            self.elements = np.load(mmap_fname+".npy", mmap_mode='r+')

    def push(self,images):
        """
        Expects images to be an ndarray of the shape B * H * W * C
        """
        if self.nelements + images.shape[0] <= self.cache_size:
            self.elements[self.nelements:self.nelements+images.shape[0]] = images
            self.nelements += images.shape[0]

        else:
            extra = self.nelements + images.shape[0] - self.cache_size

            if extra < images.shape[0]:
                self.elements[self.nelements:self.nelements + images.shape[0] - extra] = images[extra:]
                self.nelements += images.shape[0] - extra

            to_replace = self.sample(extra)
            self.elements[to_replace] = images[:extra]

    def next(self):
        """
        returns self.batch_size images
        """
        chosen = self.sample(self.batch_size)
        return self.elements[chosen]

    def sample(self,n):
        return np.random.choice(range(self.nelements),n)

if __name__ == "__main__":
    pass
    # rb = ReplayCache(100,8,[4,4,1],"mmap")
    # for i in range(50):
    #     rb.push(np.ones((8,4,4,1))*i)
    #     print(rb.elements[80])
    #     pause = input()
