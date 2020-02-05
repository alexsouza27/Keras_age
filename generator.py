from keras.utils import Sequence, to_categorical

class ValidGenerator(Sequence):
  
  def __init__(self, images_path, batch_size = 32, image_size = 64):
    self.batch_size = batch_size
    self.image_size = image_size
    self.images_path = images_path

  def __len__(self):
    return self.images // self.batch_size

  def __getitem__(self, idx):
    x = np.zeros((batch_size, image_size, image_size, 3), dtype = np.uint8)
    y = np.zeros((batch_size, 1), dtype = np.int32)

    for i in range(batch_size):
      image_path = self.images_path[idx * batch_size + i]
      image = cv2.imread(str(image_path))
      age = os.path.basename(image).split("_")[0]

      x[i] = cv2.resize(image, (self.image_size, self.image_size))
      y[i] = int(age)

    return x
