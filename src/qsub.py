from logging import getLogger
import subprocess
import tempfile
import os
import re
import time

from easydict import EasyDict as edict


default_logger = getLogger('landmark')

# rt_G.large, rt_C.large ... max=72 hours
# rt_G.small, rt_C.small ... max=168 hours
# 16分割なら rt_C.large, 8分割なら rt_C.small
JOB_TEMPLATE = """
#!/bin/bash
#$ -l {instance_type:s}=1
#$ -l h_rt={n_hours:d}:00:00
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh
module load cuda/10.0/10.0.130
module load cudnn/7.5/7.5.0
source ~/anaconda3/bin/activate landmark19 &&\
    PYTHONPATH=/fs2/groups2/gca50080/yokoo/Landmark2019-1st-and-3rd-Place-Solution {cmd_str:s}
"""

JOB_CPU_TEMPLATE = """
#!/bin/bash
#$ -l {instance_type:s}=1
#$ -l h_rt={n_hours:d}:00:00
#$ -j y
#$ -cwd
source ~/anaconda3/bin/activate landmark19 &&\
    PYTHONPATH=/fs2/groups2/gca50080/yokoo/Landmark2019-1st-and-3rd-Place-Solution {cmd_str:s}
"""


def monitor_jobs(job_ids, logger=default_logger):
    while True:
        res = check_job_running(job_ids)
        if not res.is_done and len(res.jobs) == 1:
            job = res.jobs[0]
            logger.info(
                f'>>> Job {job.job_id} ({job.name}) is running....')
        elif not res.is_done and len(res.jobs) > 1:
            logger.info(
                f'>>> {len(res.jobs)} jobs are running....')
        elif res.is_done or len(res.jobs) == 0:
            logger.info(f'>>> Job is completed!')
            break

        time.sleep(120)


def check_job_running(job_ids):
    proc = subprocess.run([
        'qstat',
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 0

    is_done = True
    res = []
    for job_id in job_ids:
        m = re.search(
            r'\s' + f'{job_id}' + r'\s+[0-9\.]+\s+([^\s]+?)\s',
            proc.stdout.decode('utf8'))
        if m is not None:
            is_done = False
            name = m.group(1)
            res.append(edict(job_id=job_id, name=name, status='runninng'))
        else:
            pass

    return edict(is_done=is_done, jobs=res)


def qsub(cmd_with_args, n_hours=1, instance_type='rt_C.small', logger=default_logger):
    job_temp = JOB_TEMPLATE
    if instance_type.startswith('rt_C'):
        job_temp = JOB_CPU_TEMPLATE

    cmd_str = " ".join(cmd_with_args)
    job_content = job_temp.strip().format(
        instance_type=instance_type,
        n_hours=n_hours,
        cmd_str=cmd_str)

    # Generate job file
    logger.info(f'Run qsub: {cmd_str}')
    tempfilename = None
    with tempfile.NamedTemporaryFile(mode='w',
                                     dir=os.getcwd(),
                                     delete=True) as f:
        f.write(job_content)
        f.flush()

        tempfilename = f.name
        proc = subprocess.run([
            'qsub',
            '-g', 'gca50080',
            tempfilename,
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc
