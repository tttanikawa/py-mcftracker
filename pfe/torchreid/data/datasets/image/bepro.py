from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset

class BeproTest(ImageDataset):
    _junk_pids = [0, -1]
    dataset_dir = 'bepro'
    # dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        # self.data_dir = self.dataset_dir
        # data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        # if osp.isdir(data_dir):
        self.data_dir = self.dataset_dir
        # else:
        #     warnings.warn(
        #         'The current data structure is deprecated. Please '
        #         'put data folders such as "bounding_box_train" under '
        #         '"Market-1501-v15.09.15".'
        #     )

        self.train_dir = osp.join(self.data_dir, 'bepro_train')
        self.query_dir = osp.join(self.data_dir, 'bepro_query')
        self.gallery_dir = osp.join(self.data_dir, 'bepro_test')
        # self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        # self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]

        # if self.market1501_500k:
            # required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        # if self.market1501_500k:
        #     gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(BeproTest, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()

        # print (dir_path)

        for img_path in img_paths:
            # pid, _ = map(int, pattern.search(img_path).groups())
            # print (img_path.split('/')[-1].split('.')[0].split('_')[0])
            # print (img_path.split('/')[0])
            # print ('pid: %s'%(img_path.split('/')[0]))
            pid = int(img_path.split('/')[-1].split('.')[0].split('_')[0])

            if pid == -1:
                continue # junk images are just ignored

            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        
        for img_path in img_paths:
            # pid, _ = map(int, pattern.search(img_path).groups())
            pid = int(img_path.split('/')[-1].split('.')[0].split('_')[0])
            camid = int(img_path.split('/')[-1].split('.')[0].split('_')[1])
            
            if pid == -1:
                continue # junk images are just ignored

            # assert 0 <= pid <= 1501 # pid == 0 means background
            # assert 1 <= camid <= 6
            # camid -= 1 # index starts from 0

            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data
