import numpy as np
import glob
from tqdm import tqdm

dir = '/work/data_and_extra/volleyball_dataset/tracked_skeletons/'

for sk in tqdm(glob.glob(dir + '**/**/**/*.npy')):
    sk_array = np.load(sk)

    if sk_array.shape != (25, 3):
        print('Trovato scheletro di shape diversa')
        # np.save(sk, np.zeros((25, 3)))

    # if np.all(sk_array == np.zeros((25, 3))):
    #     print('Trovato scheletro di zeri')

    # if sk_array.shape == (25, 3):
    #     zero_values = sk_array < 0.0001
    #     null_rows = np.any(zero_values, axis=1)
    #     print(sk_array[null_rows])
    # row = np.any(sk_array < 0.0001, axis=1)
    # print(row)
    # print(sk_array[row, :])