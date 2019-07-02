"""
    This file provides argument definitions for all experiments.
"""

def add_base_args(parser):
    parser.add_argument('--exp_name', type=str, default='no_name', help='name of the training run')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility)')
    parser.add_argument('--deterministic', action='store_true', default=False, help='set pytorch and cudnn to deterministic mode (slower but should be fully deterministic)')
    return parser

def add_model_args(parser):
    parser.add_argument('--model_path', type=str, default='../data/models')
    return parser

def add_data_args(parser):
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='train.txt', help='file name for the list of object names')
    parser.add_argument('--max_data_tree_depth', type=int, default=-1, help='maximum hierarchy tree depth, prune above this depth')
    parser.add_argument('--edge_types', type=str, nargs='*', default=['ADJ', 'ROT_SYM', 'TRANS_SYM', 'REF_SYM'], help='list of possible edge types')
    return parser

def add_train_vae_args(parser):
    parser = add_base_args(parser)
    parser = add_model_args(parser)
    parser = add_data_args(parser)

    # validation dataset
    parser.add_argument('--val_dataset', type=str, default='val.txt', help='file name for the list of validation object names')

    # model hyperparameters
    parser.add_argument('--box_code_size', type=int, default=12)
    parser.add_argument('--geo_feat_size', type=int, default=100)
    parser.add_argument('--feature_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--edge_feature_size', type=int, default=0, help='Dimension of the edge features (parameters). Set to 0 if edge features are not used.')
    parser.add_argument('--edge_dec_hidden_size', type=int, default=256, help='In the decoder, edge features are decoded from pairs of node features. This is the dimension of these node features.')
    parser.add_argument('--num_groups', type=int, default=None)
    parser.add_argument('--num_point', type=int, default=1000)
    parser.add_argument('--load_geo', action='store_true', default=False)
    parser.add_argument('--load_geo_feat', action='store_true', default=False)
    parser.add_argument('--symmetry_size', type=int, default=8)
    # parser.add_argument('--max_box_num', type=int, default=30)
    # parser.add_argument('--max_sym_num', type=int, default=10)
    parser.add_argument('--max_tree_depth', type=int, default=100, help='maximum depth of generated object trees')
    parser.add_argument('--max_child_num', type=int, default=10, help='maximum number of children per parent')
    parser.add_argument('--child_encoder_type', type=str, default='concat', help='type of child encoder')
    parser.add_argument('--symmetric_type', type=str, default='max', help='node pooling type')
    parser.add_argument('--edge_symmetric_type', type=str, default='max', help='edge pooling type')
    parser.add_argument('--dec_edge_symmetric_type', type=str, default='max', help='edge pooling type')
    parser.add_argument('--child_decoder_type', type=str, default='concat', help='type of child decoder')
    parser.add_argument('--num_gnn_iterations', type=int, default=2, help='number of message passing iterations for the GNN')
    parser.add_argument('--num_dec_gnn_iterations', type=int, default=2, help='number of message passing iterations for the GNN')
    parser.add_argument('--shared_gnn_iterations', action='store_true', default=False, help='share encoders between iterations')
    parser.add_argument('--shared_dec_gnn_iterations', action='store_true', default=False, help='share encoders between iterations')
    parser.add_argument('--edge_feat_type', type=str, default='type_only', help='edge feature type')
    parser.add_argument('--model_version', type=str, default='grass', help='model version, this is the suffix in models_xxx.py')

    # training parameters
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=500)
    parser.add_argument('--optimizer', type=str, default='adam', help='which optimizer')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training run')
    parser.add_argument('--non_variational', action='store_true', default=False, help='make the variational autoencoder non-variational')
    parser.add_argument('--no_fold', action='store_true', default=False, help='dont use torchfold to batch computations')

    # loss weights
    parser.add_argument('--loss_weight_geo', type=float, default=1.0, help='weight for the geo recon loss')
    parser.add_argument('--loss_weight_latent', type=float, default=1.0, help='weight for the latent recon loss')
    parser.add_argument('--loss_weight_center', type=float, default=1.0, help='weight for the center recon loss')
    parser.add_argument('--loss_weight_scale', type=float, default=1.0, help='weight for the scale recon loss')
    parser.add_argument('--loss_weight_sym', type=float, default=50.0, help='weight for the sym loss')
    parser.add_argument('--loss_weight_adj', type=float, default=50.0, help='weight for the adj loss')
    parser.add_argument('--loss_weight_kldiv', type=float, default=0.05, help='weight for the kl divergence loss')
    parser.add_argument('--loss_weight_box', type=float, default=1.0, help='weight for the box reconstruction loss')
    parser.add_argument('--loss_weight_anchor', type=float, default=1.0, help='weight for the anchor reconstruction loss')
    parser.add_argument('--loss_weight_leaf', type=float, default=1.0, help='weight for the "node is leaf" reconstruction loss')
    parser.add_argument('--loss_weight_exists', type=float, default=1.0, help='weight for the "node exists" reconstruction loss')
    parser.add_argument('--loss_weight_semantic', type=float, default=1.0, help='weight for the semantic reconstruction loss')
    parser.add_argument('--loss_weight_edge_exists', type=float, default=1.0, help='weight for the "edge exists" loss')
    parser.add_argument('--loss_weight_edge_feats', type=float, default=1.0, help='weight for the edge feature loss')

    # logging
    parser.add_argument('--log_path', type=str, default='../data/logs')
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=3, help='number of optimization steps beween console log prints')
    parser.add_argument('--checkpoint_interval', type=int, default=500, help='number of optimization steps beween checkpoints')
    parser.add_argument('--checkpoint_epoch_interval', type=int, default=20, help='number of optimization epochs beween checkpoints')

    # resume
    parser.add_argument('--resume_from_another_exp', action='store_true', default=False, help='resume from another exp')
    parser.add_argument('--resume_ckpt_dir', type=str, help='resume ckpt dir')
    parser.add_argument('--resume_model_epoch', type=int, help='resume model epoch')
    parser.add_argument('--pc_ae_model_epoch', type=int, help='resume model epoch')
    parser.add_argument('--pc_ae_name', type=str, help='resume model epoch')

    return parser

def add_train_vae_gan_args(parser):
    parser = add_train_vae_args(parser)

    parser.add_argument('--pretrained_vae_name', type=str, default='')
    parser.add_argument('--pretrained_vae_epoch', type=int, default=-1, help='vae at what epoch to use (set to < 0 for the final/most recent epoch)')

    parser.add_argument('--loss_weight_gan', type=float, default=1.0, help='weight for the gan loss')

    return parser

def add_train_bbox_vae_args(parser):
    parser = add_train_vae_args(parser)

    parser.add_argument('--max_box_num', type=int, default=100)
    parser.add_argument('--encoder_type', type=str, default='mlp', help='mlp or pointnet')
    parser.add_argument('--bbox_matching_type', type=str, default='feats', help='match bounding boxes in output and ground truth based on "feats" (full box features) or "center" (box center only)')

    return parser

def add_train_graph_vae_args(parser):
    parser = add_train_vae_args(parser)

    parser.add_argument('--max_node_num', type=int, default=20)
    parser.add_argument('--encoder_type', type=str, default='mlp', help='mlp or pointnet')
    parser.add_argument('--graph_matching_type', type=str, default='node_center', help='match bounding boxes in output and ground truth based on "feats" (full box features) or "center" (box center only)')
    parser.add_argument('--use_labels', action='store_true', default=False)

    return parser

def add_result_args(parser):
    parser = add_base_args(parser)
    parser = add_model_args(parser)

    parser.add_argument('--result_path', type=str, default='../data/results')
    parser.add_argument('--model_epoch', type=int, default=-1, help='model at what epoch to use (set to < 0 for the final/most recent model)')

    return parser

def add_gen_args(parser):
    parser = add_result_args(parser)

    parser.set_defaults(result_path='../data/results/gen')

    parser.add_argument('--gen_count', type=int, default=10, help='number of generated objects')

    return  parser

def add_eval_args(parser):
    parser = add_result_args(parser)
    parser = add_data_args(parser)

    parser.set_defaults(data_path='') # empty means use the training config
    parser.set_defaults(data_type='') # empty means use the training config
    parser.set_defaults(result_path='../data/results/eval')

    return  parser

def add_shape_pts_to_partnetobb_args(parser):
    parser.add_argument('--decoder_name', type=str)
    parser.add_argument('--decoder_model_version', type=str)
    parser.add_argument('--decoder_model_epoch', type=int)
    parser.add_argument('--partnetobb_data_path', type=str)
    return  parser

def add_interpolation_args(parser):
    parser.add_argument('--shape_a', type=str)
    parser.add_argument('--shape_b', type=str)
    parser.add_argument('--num_interps', type=int, default=10)
    parser.add_argument('--interp_sem', type=str, default=None)
    return  parser

def add_free_generation_args(parser):
    parser.add_argument('--num_shapes', type=int, default=100)
    return  parser

def add_editing_args(parser):
    parser.add_argument('--anno_id', type=str)
    parser.add_argument('--part_id', type=int)
    parser.add_argument('--ori_part_id', type=int)
    parser.add_argument('--new_box_params', type=str)
    parser.add_argument('--lr', type=int, default=1e-1)
    parser.add_argument('--lr_decay_by', type=float, default=0.8)
    parser.add_argument('--lr_decay_every', type=float, default=100)
    parser.add_argument('--max_iterations', type=int, default=101)
    return  parser

def add_multicpu_args(parser):
    parser.add_argument('--num_processes', type=int, default=5, help='number of processes to parallel training')
    parser.add_argument('--num_epochs_per_process', type=int, default=200, help='number of epochs per process')

    return parser
