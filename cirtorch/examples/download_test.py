import os
from cirtorch.utils.download import download_test
from cirtorch.utils.general import get_root

data_root = os.path.join(get_root(), 'cirtorch')
download_test(data_root)
