from logging import getLogger
import contextlib
import time


logger = getLogger('landmark')


@contextlib.contextmanager
def timer(name, disable_log=False):
    st_time = time.time()
    yield
    if not disable_log:
        logger.info(f'[{name}] done in {time.time() - st_time:.2f} s')
