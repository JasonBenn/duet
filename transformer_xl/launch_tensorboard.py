import os
import ncluster
task = ncluster.make_task('tensorboard', instance_type='r5.large',
                          image_name='Deep Learning AMI (Ubuntu) Version 22.0')

task.run('source activate tensorflow_p36')
logdir_root = os.path.dirname(task.logdir)
task.run(f'tensorboard --logdir={ncluster.get_logdir_root()} --port=6006',
         non_blocking=True)
print(f'TensorBoard at http://{task.public_ip}:6006')
