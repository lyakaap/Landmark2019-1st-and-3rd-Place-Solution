from logging import getLogger, INFO
import re
import time
from pathlib import Path

import click

from util import init_cli
import qsub


logger = getLogger('landmark')
logger.setLevel(INFO)


@click.group()
def cli():
    init_cli(logger)


@cli.command()
def train19_train19_same_landmark_search():
    __train19_train19_same_landmark_search()


@cli.command()
def train19_train19_same_landmark_verify():
    __train19_train19_same_landmark_verify()


@cli.command()
def extract_delf_v190505_test():
    __extract_delf_v190505_test()


@cli.command()
def extract_delf_v190505_index():
    __extract_delf_v190505_index()


@cli.command()
def extract_delf_v190505_train19():
    __extract_delf_v190505_train19()


@cli.command()
def extract_delf_v190505_index19():
    __extract_delf_v190505_index19()


@cli.command()
def extract_delf_v190505_test19():
    __extract_delf_v190505_test19()


def __extract_delf_v190505_index19():
    Path('data/input/delf_190505/index19').mkdir(
        parents=True, exist_ok=True)

    job_ids = []
    for blockid in range(2, 32+1):
        cmd_with_args = [
            "python", "-W", "ignore",
            "code/exp32/delf_extract_features.py",
            "--config_path",
            "code/exp32/params/delf_config.pbtxt",
            "--list_images_path",
            f"data/working/index19.lst_blk{blockid}",
            "--output_dir",
            "data/input/delf_190505/index19/",
        ]
        proc = qsub.qsub(cmd_with_args,
                         n_hours=5,
                         instance_type='rt_G.small')
        logger.info(f'Response from qsub: {proc.returncode}')

        m = re.match(r'Your job (\d+) \(', proc.stdout.decode('utf8'))
        job_id = int(m.group(1)) if m is not None else None
        logger.info(f'Job id: {job_id}')
        assert job_id is not None
        job_ids.append(job_id)
        time.sleep(1)

    qsub.monitor_jobs(job_ids)


def __extract_delf_v190505_test19():
    Path('data/input/delf_190505/test19').mkdir(
        parents=True, exist_ok=True)

    job_ids = []
    for blockid in range(1, 16+1):
        cmd_with_args = [
            "python", "-W", "ignore",
            "code/exp32/delf_extract_features.py",
            "--config_path",
            "code/exp32/params/delf_config.pbtxt",
            "--list_images_path",
            f"data/working/test19.lst_blk{blockid}",
            "--output_dir",
            "data/input/delf_190505/test19/",
        ]
        proc = qsub.qsub(cmd_with_args,
                         n_hours=3,
                         instance_type='rt_G.small')
        logger.info(f'Response from qsub: {proc.returncode}')

        m = re.match(r'Your job (\d+) \(', proc.stdout.decode('utf8'))
        job_id = int(m.group(1)) if m is not None else None
        logger.info(f'Job id: {job_id}')
        assert job_id is not None
        job_ids.append(job_id)
        time.sleep(1)

    qsub.monitor_jobs(job_ids)


def __train19_train19_same_landmark_verify():
    job_ids = []
    for blockid in range(1, 32+1):
        cmd_with_args = [
            "python", "-W", "ignore",
            "code/exp12/main.py",
            "prep-delf-results",
            "-b", f"{blockid}",
        ]
        proc = qsub.qsub(cmd_with_args,
                         n_hours=22,
                         instance_type='rt_C.large')
        logger.info(f'Response from qsub: {proc.returncode}')

        m = re.match(r'Your job (\d+) \(', proc.stdout.decode('utf8'))
        job_id = int(m.group(1)) if m is not None else None
        logger.info(f'Job id: {job_id}')
        assert job_id is not None
        job_ids.append(job_id)
        time.sleep(1)

    qsub.monitor_jobs(job_ids)


def __train19_train19_same_landmark_search():
    job_ids = []
    for blockid in range(2, 32+1):
        cmd_with_args = [
            "python", "-W", "ignore",
            "code/exp12/main.py",
            "prep-faiss-search-results",
            "-b", f"{blockid}",
        ]
        proc = qsub.qsub(cmd_with_args,
                         n_hours=2,
                         instance_type='rt_G.small')
        logger.info(f'Response from qsub: {proc.returncode}')

        m = re.match(r'Your job (\d+) \(', proc.stdout.decode('utf8'))
        job_id = int(m.group(1)) if m is not None else None
        logger.info(f'Job id: {job_id}')
        assert job_id is not None
        job_ids.append(job_id)
        time.sleep(1)

    qsub.monitor_jobs(job_ids)


def __extract_delf_v190505_test():
    job_ids = []
    cmd_with_args = [
        "python", "-W", "ignore",
        "code/exp12/delf_extract_features.py",
        "--config_path",
        "code/exp12/params/delf_config.pbtxt",
        "--list_images_path",
        f"data/working/exp12/test_filepath.lst",
        "--output_dir",
        "data/input/delf_190505/test/",
    ]
    proc = qsub.qsub(cmd_with_args,
                     n_hours=12,
                     instance_type='rt_G.small')
    logger.info(f'Response from qsub: {proc.returncode}')

    m = re.match(r'Your job (\d+) \(', proc.stdout.decode('utf8'))
    job_id = int(m.group(1)) if m is not None else None
    logger.info(f'Job id: {job_id}')
    assert job_id is not None
    job_ids.append(job_id)
    time.sleep(1)

    qsub.monitor_jobs(job_ids)


def __extract_delf_v190505_index():
    imlist = []
    with open('data/working/index_exists.lst', 'r') as f:
        for name in f:
            name = name.strip()
            fn = f'data/input/index_r1024/{name}.jpg'
            imlist.append(fn)

    for blockid in range(1, 32+1):
        fn_list = f"data/working/index_exists.lst_blk{blockid}.lst"
        with open(fn_list, 'w') as f:
            for i in range(len(imlist)):
                if i % 32 == (blockid - 1):
                    f.write(imlist[i] + '\n')

    Path('data/input/delf_190505/index').mkdir(parents=True, exist_ok=True)

    job_ids = []
    for blockid in range(1, 32+1):
        cmd_with_args = [
            "python", "-W", "ignore",
            "code/exp12/delf_extract_features.py",
            "--config_path",
            "code/exp12/params/delf_config.pbtxt",
            "--list_images_path",
            f"data/working/index_exists.lst_blk{blockid}.lst",
            "--output_dir",
            "data/input/delf_190505/index/",
        ]
        proc = qsub.qsub(cmd_with_args,
                         n_hours=12,
                         instance_type='rt_G.small')
        logger.info(f'Response from qsub: {proc.returncode}')

        m = re.match(r'Your job (\d+) \(', proc.stdout.decode('utf8'))
        job_id = int(m.group(1)) if m is not None else None
        logger.info(f'Job id: {job_id}')
        assert job_id is not None
        job_ids.append(job_id)
        time.sleep(1)

    qsub.monitor_jobs(job_ids)


def __extract_delf_v190505_train19():
    job_ids = []
    for blockid in range(1, 32+1):
        cmd_with_args = [
            "python", "-W", "ignore",
            "code/exp12/delf_extract_features.py",
            "--config_path",
            "code/exp12/params/delf_config.pbtxt",
            "--list_images_path",
            f"data/working/train19_exists.lst_blk{blockid}.lst",
            "--output_dir",
            "data/input/delf_190505/train19/",
        ]
        proc = qsub.qsub(cmd_with_args,
                         n_hours=12,
                         instance_type='rt_G.small')
        logger.info(f'Response from qsub: {proc.returncode}')

        m = re.match(r'Your job (\d+) \(', proc.stdout.decode('utf8'))
        job_id = int(m.group(1)) if m is not None else None
        logger.info(f'Job id: {job_id}')
        assert job_id is not None
        job_ids.append(job_id)
        time.sleep(1)

    qsub.monitor_jobs(job_ids)


if __name__ == '__main__':
    cli()
