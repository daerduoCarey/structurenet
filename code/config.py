"""
    This file provides argument definitions for all experiments.
"""

def add_base_args(parser):
    parser.add_argument('--exp_name', type=str, default='no_name', help='name of the training run')
    parser.add_argument('--category', type=str, default='Chair', help='object category')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility)')

    return parser

def add_model_args(parser):
    parser.add_argument('--model_path', type=str, default='../data/models')

    return parser

def add_data_args(parser):
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--train_dataset', type=str, default='train.txt', help='file name for the list of object names for training')
    parser.add_argument('--edge_types', type=str, nargs='*', default=['ADJ', 'ROT_SYM', 'TRANS_SYM', 'REF_SYM'], help='list of possible edge types')

    return parser

def add_train_vae_args(parser):
    parser = add_base_args(parser)
    parser = add_model_args(parser)
    parser = add_data_args(parser)

    # validation dataset
    parser.add_argument('--val_dataset', type=str, default='val.txt', help='file name for the list of object names for validation')

    # model hyperparameters
    parser.add_argument('--geo_feat_size', type=int, default=100)
    parser.add_argument('--feature_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_point', type=int, default=1000)
    parser.add_argument('--load_geo', action='store_true', default=False)
    parser.add_argument('--max_tree_depth', type=int, default=100, help='maximum depth of generated object trees')
    parser.add_argument('--max_child_num', type=int, default=10, help='maximum number of children per parent')
    parser.add_argument('--node_symmetric_type', type=str, default='max', help='node pooling type')
    parser.add_argument('--edge_symmetric_type', type=str, default='avg', help='edge pooling type')
    parser.add_argument('--num_gnn_iterations', type=int, default=2, help='number of message passing iterations for the GNN encoding')
    parser.add_argument('--num_dec_gnn_iterations', type=int, default=2, help='number of message passing iterations for the GNN decoding')
    parser.add_argument('--model_version', type=str, default='model', help='model file name')

    # training parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=500)
    parser.add_argument('--non_variational', action='store_true', default=False, help='make the variational autoencoder non-variational')

    # loss weights
    parser.add_argument('--loss_weight_geo', type=float, default=2.0, help='weight for the geo recon loss')
    parser.add_argument('--loss_weight_latent', type=float, default=20.0, help='weight for the latent recon loss')
    parser.add_argument('--loss_weight_center', type=float, default=20.0, help='weight for the center recon loss')
    parser.add_argument('--loss_weight_scale', type=float, default=20.0, help='weight for the scale recon loss')
    parser.add_argument('--loss_weight_sym', type=float, default=1.0, help='weight for the sym loss')
    parser.add_argument('--loss_weight_adj', type=float, default=1.0, help='weight for the adj loss')
    parser.add_argument('--loss_weight_kldiv', type=float, default=0.05, help='weight for the kl divergence loss')
    parser.add_argument('--loss_weight_box', type=float, default=20.0, help='weight for the box reconstruction loss')
    parser.add_argument('--loss_weight_anchor', type=float, default=10.0, help='weight for the anchor reconstruction loss')
    parser.add_argument('--loss_weight_leaf', type=float, default=1.0, help='weight for the "node is leaf" reconstruction loss')
    parser.add_argument('--loss_weight_exists', type=float, default=1.0, help='weight for the "node exists" reconstruction loss')
    parser.add_argument('--loss_weight_semantic', type=float, default=0.1, help='weight for the semantic reconstruction loss')
    parser.add_argument('--loss_weight_edge_exists', type=float, default=1.0, help='weight for the "edge exists" loss')

    # logging
    parser.add_argument('--log_path', type=str, default='../data/logs')
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=3, help='number of optimization steps beween console log prints')
    parser.add_argument('--checkpoint_interval', type=int, default=500, help='number of optimization steps beween checkpoints')

    # load pretrained model (for pc exps)
    parser.add_argument('--part_pc_exp_name', type=str, help='resume model exp name')
    parser.add_argument('--part_pc_model_epoch', type=int, help='resume model epoch')

    return parser

def add_result_args(parser):
    parser = add_base_args(parser)
    parser = add_model_args(parser)

    parser.add_argument('--result_path', type=str, default='../data/results')
    parser.add_argument('--model_epoch', type=int, default=-1, help='model at what epoch to use (set to < 0 for the final/most recent model)')

    return parser

def add_eval_args(parser):
    parser = add_result_args(parser)
    parser = add_data_args(parser)
    
    parser.add_argument('--test_dataset', type=str, default='test.txt', help='file name for the list of object names for testing')

    return  parser

