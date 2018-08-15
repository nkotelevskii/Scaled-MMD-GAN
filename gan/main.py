import sys
from utils.misc import pp, visualize
import yaml
import tensorflow as tf
import argparse
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_flags(parser):
    FLAGS = parser.parse_args()
    if FLAGS.config_file:
        config = yaml.load(open(FLAGS.config_file))
        dic = vars(FLAGS)
        all(map(dic.pop, config))
        dic.update(config)
    return FLAGS


parser = argparse.ArgumentParser()


def add_arg(name, **kwargs):
    "Convenience to handle reasonable names for args as well as crappy ones."
    assert name[0] == '-'
    assert '-' not in name[1:]
    nice_name = '--' + name[1:].replace('_', '-')
    return parser.add_argument(name, nice_name, **kwargs)

# Optimizer
add_arg('-max_iteration',               default=150000,         type=int,       help='Epoch to train [400000]')
add_arg('-beta1',                       default=0.5,            type=float,     help='Momentum term of adam [0.5]')
add_arg('-beta2',                       default=0.9,            type=float,     help='beta2 in adam [0.9]')
add_arg('-learning_rate',               default=0.0001,         type=float,     help='Learning rate [2]')
add_arg('-learning_rate_D',             default=-1,             type=float,     help='Learning rate for discriminator, if negative same as generator [-1]')
add_arg('-dsteps',                      default=5,              type=int,       help='Number of discriminator steps in a row [1]')
add_arg('-gsteps',                      default=1,              type=int,       help='Number of generator steps in a row [1]')
add_arg('-start_dsteps',                default=10,             type=int,       help='Number of discrimintor steps in a row during first 20 steps and every 100th step [1]')

add_arg('-clip_grad',                   default=True,           type=str2bool,  help='Use gradient clippint [True]')
add_arg('-batch_norm',                  default=False,          type=str2bool,  help='Use of batch norm [False]')

# Initalization params
add_arg('-init',                        default=0.02,           type=float,     help='Initialization value [0.02]')
# dimensions
add_arg('-batch_size',                  default=64,             type=int,       help='The size of batch images [1000]')
add_arg('-real_batch_size',             default=-1,            type=int,       help='The size of batch images for real samples. If -1 then same as batch_size [-1]')
add_arg('-output_size',                 default=128,            type=int,       help='The size of the output images to produce [64]')
add_arg('-c_dim',                       default=3,              type=int,       help='Dimension of image color. [3]')
add_arg('-z_dim',                       default=128,            type=int,       help='Dimension of latent noise [128]')
add_arg('-df_dim',                      default=64,             type=int,       help='Discriminator no of channels at first conv layer [64]')
add_arg('-dof_dim',                     default=1,              type=int,       help='No of discriminator output features [1]')
add_arg('-gf_dim',                      default=64,             type=int,       help='No of generator channels [64]')
# Directories
add_arg('-dataset',                     default="cifar10",      type=str,       help='The name of the dataset [celebA, mnist, lsun, cifar10, imagenet]')
add_arg('-name',                        default="",             type=str,       help='The name of the experiment for saving purposes ')
add_arg('-checkpoint_dir',              default="checkpoint",   type=str,       help='Directory name to save the checkpoints [checkpoint_mmd]')
add_arg('-sample_dir',                  default="sample",       type=str,       help='Directory name to save the image samples [samples_mmd]')
add_arg('-log_dir',                     default="log",          type=str,       help='Directory name to save the image samples [logs_mmd]')
add_arg('-data_dir',                    default="data",         type=str,       help='Directory containing datasets [./data]')
add_arg('-out_dir',                     default="",             type=str,       help='Directory name to save the outputs of the experiment : (log, sample, checkpoints)   [./data]')
add_arg('-config_file',                 default="",             type=str,       help='path to the config file')

# models
add_arg('-architecture',                default="dcgan",        type=str,       help='The name of the architecture [dcgan, g-resnet5, dcgan5]')
add_arg('-kernel',                      default="",             type=str,       help="The name of the kernel ['', 'mix_rbf', 'mix_rq', 'distance', 'dot', 'mix_rq_dot']")
add_arg('-model',                       default="mmd",          type=str,       help='The model type [mmd, smmd, swgan, wgan_gp]')

# training options
add_arg('-is_train',                    default=True,           type=str2bool,  help='True for training, False for testing [Train]')
add_arg('-visualize',                   default=False,          type=str2bool,  help='True for visualizing, False for nothing [False]')
add_arg('-is_demo',                     default=False,          type=str2bool,  help='For testing [False]')

add_arg('-log',                         default=True,           type=str2bool,  help='Wheather to write log to a file in samples directory [True]')
add_arg('-compute_scores',              default=True,           type=str2bool,  help='Compute scores')
add_arg('-print_pca',                   default=False,          type=str2bool,  help='Print the PCA')
add_arg('-suffix',                      default="",             type=str,       help="Fo additional settings ['', '_tf_records']")
add_arg('-gpu_mem',                     default=.9,             type=float,     help="GPU memory fraction limit [1.0]")
add_arg('-no_of_samples',               default=100000,         type=int,       help="number of samples to produce")
add_arg('-save_layer_outputs',          default=0,              type=int,       help="Wheather to save_layer_outputs. If == 2, saves outputs at exponential steps: 1, 2, 4, ..., 512 and every 1000. [0, 1, 2]")
add_arg('-ckpt_name',                   default="",             type=str,       help=" Name of the checkpoint to load ")

# Decay rates
add_arg('-decay_rate',                  default=.8,             type=float,     help='Decay rate [1.0]')
add_arg('-gp_decay_rate',               default=.8,             type=float,     help='Decay rate of the gradient penalty [1.0]')
add_arg('-sc_decay_rate',               default=1.,             type=float,     help='Decay of the scaling factor')
add_arg('-restart_lr',                  default=False,          type=str2bool,  help='Wheather to use lr scheduler based on 3-sample test')
add_arg('-restart_sc',                  default=False,          type=str2bool,  help='Ensures the discriminator network is injective by adding the input to the feature')
add_arg('-MMD_lr_scheduler',            default=True,           type=str2bool,  help='Whether to use lr scheduler based on 3-sample test')
add_arg('-MMD_sdlr_past_sample',        default=10,             type=int,       help='lr scheduler: number of past iterations to keep')
add_arg('-MMD_sdlr_num_test',           default=3,              type=int,       help='lr scheduler: number of failures to decrease KID score')
add_arg('-MMD_sdlr_freq',               default=2000,           type=int,       help='lr scheduler: frequency of scoring the model')

# discriminator penalties
add_arg('-gradient_penalty',            default=0.0,            type=float,     help='Use gradient penalty if > 0.0 [0.0]')
add_arg('-L2_discriminator_penalty',    default=0.0,            type=float,     help="Use L2 penalty on discriminator features if > 0.0 [0.0]")

# scaled MMD
add_arg('-with_scaling',                default=False,          type=str2bool,  help='Use scaled MMD')
add_arg('-scaling_coeff',               default=10.,            type=float,     help='coeff of scaling')
add_arg('-scaling_variant',             default='grad',         type=str,       help='The variant of the scaled MMD   [value_and_grad, grad]')
add_arg('-use_gaussian_noise',          default=False,          type=str2bool,  help='Add N(0, 10^2) noise to images in scaling')

# spectral normalization
add_arg('-with_sn',                     default=False,          type=str2bool,  help='use spectral normalization')
add_arg('-with_learnable_sn_scale',     default=False,          type=str2bool,  help='train the scale of normalized weights')

# incomplete cholesky options for sobolevmmd
add_arg('-use-incomplete-cho',          default=True,           type=str2bool,  help="whether to use incomplete Cholesky for sobolevmmd [true]")
add_arg('-incho-eta',                   default=1e-3,           type=float,     help="stopping criterion for incomplete cholesky [%(default)s]")
add_arg('-incho-max-steps',             default=1000,           type=int,       help="iteration cap for incomplete cholesky [%(default)s]")

# multi-gpu training
add_arg('-multi_gpu',                   default=False,          type=str2bool,  help=' train accross multiple gpus in a multi-tower fashion')
add_arg('-num_gpus',                    default=1,              type=int,       help='Number of GPUs to use')
# conditional gan, only for imagenet
add_arg('-with_labels',                 default=False,          type=str2bool,  help='Conditional GAN')


FLAGS = make_flags(parser)

FLAGS.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


def main(_):
    pp.pprint(vars(FLAGS))

    sess_config = tf.ConfigProto(
        device_count={"CPU": 3},
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    if FLAGS.model == 'mmd':
        from core.model import MMD_GAN as Model
    elif FLAGS.model == 'gan':
        from core.gan import GAN as Model
    elif FLAGS.model == 'wgan_gp':
        from core.wgan_gp import WGAN_GP as Model
    elif FLAGS.model == 'cramer':
        from core.cramer import Cramer_GAN as Model
    elif FLAGS.model == 'smmd':
        from core.smmd import SMMD as Model
    elif FLAGS.model == 'swgan':
        from core.smmd import SWGAN as Model
    else:
        raise ValueError("unknown model {}".format(FLAGS.model))

    #if FLAGS.multi_gpu:
    #    from core.model_multi_gpu import MMD_GAN as Model
    with tf.Session(config=sess_config) as sess:
        #sess = tf_debug.tf_debug.TensorBoardDebugWrapperSession(sess,'localhost:6064')
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        if FLAGS.dataset == 'mnist':
            gan = Model(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=28, c_dim=1,
                        data_dir=FLAGS.data_dir)
        elif FLAGS.dataset == 'cifar10':
            gan = Model(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=32, c_dim=3,
                        data_dir=FLAGS.data_dir)
        elif FLAGS.dataset in ['celebA', 'lsun', 'imagenet']:
            gan = Model(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=FLAGS.output_size, c_dim=3,
                        data_dir=FLAGS.data_dir)
        else:
            gan = Model(
                sess, batch_size=FLAGS.batch_size,
                output_size=FLAGS.output_size, c_dim=FLAGS.c_dim,
                data_dir=FLAGS.data_dir)

        if FLAGS.is_train:
            gan.train()
            gan.pre_process_only()
        elif FLAGS.print_pca:
            gan.print_pca()
        elif FLAGS.visualize:
            gan.load_checkpoint()
            visualize(sess, gan, FLAGS, 2)
        else:
            gan.get_samples(FLAGS.no_of_samples, layers=[-1])

        if FLAGS.log:
            sys.stdout = gan.old_stdout
            gan.log_file.close()
        gan.sess.close()

if __name__ == '__main__':
    tf.app.run()
