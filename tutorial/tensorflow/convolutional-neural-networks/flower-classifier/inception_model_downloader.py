import os
import sys
import re
import tarfile
from pathlib import Path
from six.moves import urllib

class InceptionModelDownloader:
    def __init__(self, url=None, path=None, class_label_url=None, checkpoint_path=None):
        if url is None:
            tf_models_repo = 'http://download.tensorflow.org/models'
            url = os.path.join(tf_models_repo, 'inception_v3_2016_08_28.tar.gz')
        if path is None:
            repo_root_dir = os.path.join(Path.home(), 'workspace/machine-learning')
            path = os.path.join(repo_root_dir, 'pretrained-models/inception')
        if class_label_url is None:
            class_label_url = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/inception/imagenet_class_names.txt'
        if checkpoint_path is None:
            checkpoint_path = os.path.join(path, 'inception_v3.ckpt')

        self.url = url
        self.path = path
        self.class_label_url = class_label_url
        self.checkpoint_path = checkpoint_path

    def _download_progress(self, count, block_size, total_size):
        percent = count * block_size * 100 // total_size
        sys.stdout.write('\rDownloading: {:2.2f}%'.format(percent))
        sys.stdout.flush()
        
    def fetch_inception_v3(self):
        if os.path.exists(self.checkpoint_path):
            return
        
        os.makedirs(self.path, exist_ok=True)

        tgz_path = os.path.join(self.path, 'inception_v3.tgz')
        urllib.request.urlretrieve(self.url, tgz_path, reporthook=self._download_progress)
        
        with tarfile.open(tgz_path) as inception_tgz:
            inception_tgz.extractall(path=self.path)
        os.remove(tgz_path)
    
    def load_class_labels(self):
        class_labels_path = os.path.join(self.path, 'imagenet_class_names.txt')
        if os.path.exists(class_labels_path) == False:
            urllib.request.urlretrieve(self.class_label_url, class_labels_path)

        with open(class_labels_path, 'rb') as file:
            content = file.read().decode('utf-8')
            # Regex mean
            # ^ is begging point. And % is end point.
            # ^n is that the sequence begin with n.
            # *$ is that the seuqnce end with any character without newline.
            # n{digits}{white-space}{any characters}{white-space}
            class_label_regex = re.compile(r'^n\d+\s+(.*)\s*$', re.M | re.U)
            return class_label_regex.findall(content)