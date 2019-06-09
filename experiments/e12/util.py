from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO
from pathlib import Path
import time
import sys
import datetime
import contextlib

# import _jsonnet


LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'


logger = getLogger('landmark')


@contextlib.contextmanager
def timer(name, disable_log=False):
    st_time = time.time()
    yield
    if not disable_log:
        logger.info(f'[{name}] done in {time.time() - st_time:.2f} s')


# def jsonnet_load(fn):
#     json_str = _jsonnet.evaluate_file(fn)
#     return edict(json.loads(json_str))


# def jsonnet_loads(snippet):
#     return edict(json.loads(_jsonnet.evaluate_snippet('snippet', snippet)))


def init_cli(logger_):
    paths = [
        'data/working/logs',
        'data/working/models',
    ]
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter(LOGFORMAT))
    logger_.addHandler(handler)

    suffix_name = Path(sys.argv[0]).name
    if len(sys.argv) > 1:
        suffix_name += '_' + sys.argv[1]
    dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')[2:]

    fn = f'data/working/logs/{dt}_{suffix_name}.log'
    fh_handler = FileHandler(fn)
    fh_handler.setFormatter(Formatter(LOGFORMAT))
    logger_.addHandler(fh_handler)
    logger_.info(f'Logfile: {fn}')
    logger_.info(f'Command: {" ".join(sys.argv)}')
    fh_handler.flush()
