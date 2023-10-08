import argparse

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_motion_file', type=str, default="./data/train_motion.npy", help='motion files')
    parser.add_argument('--train_image_file', type=str, default="./data/train_image.npy", help='image embedding')
    parser.add_argument('--val_motion_file', type=str,
                        default="./data/val_motion.npy",
                        help='motion files')
    parser.add_argument('--val_image_file', type=str,
                        default="./data/val_image.npy",
                        help='image embedding')

    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    # parser.add_argument('--lr', type=float, default=3e-5, help='input the learning rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='input the learning rate')
    parser.add_argument('--dim_pose', type=int, default=3, help='input of motion')
    parser.add_argument('--joints_num', type=int, default=16, help='input of motion')
    # parser.add_argument('--lr', type=float, required=True, default=3e-5, help='input the learning rate')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=3000)
    parser.add_argument('--negative_margin', type=float, default=1)

    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--is_continue', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--log_dir', default='/home/cjm/CALM/Anyskill/log/')
    parser.add_argument('--model_dir', default='/home/cjm/CALM/Anyskill/output/')
    parser.add_argument('--eval_dir', default='/home/cjm/CALM/Anyskill/eval/')

    parser.add_argument('--dim_movement_enc_hidden', type=int, default=512, help='Dimension of hidden unit in GRU')
    parser.add_argument('--dim_movement_latent', type=int, default=512, help='Dimension of hidden unit in GRU')
    parser.add_argument('--dim_motion_hidden', type=int, default=1024, help='Dimension of hidden unit in GRU')
    parser.add_argument('--dim_coemb_hidden', type=int, default=512, help='Dimension of hidden unit in GRU')

    parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
    parser.add_argument('--save_every_e', type=int, default=50, help='Frequency of printing training progress')
    parser.add_argument('--eval_every_e', type=int, default=30, help='Frequency of printing training progress')
    parser.add_argument('--save_latest', type=int, default=50, help='Frequency of printing training progress')

    opt = parser.parse_args()

    return opt

def test_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_motion_file', type=str, default="./data/val_motion.npy", help='motion files')
    parser.add_argument('--test_image_file', type=str, default="./data/val_image.npy", help='image embedding')

    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='input the learning rate')
    parser.add_argument('--dim_pose', type=int, default=13, help='input of motion')
    parser.add_argument('--joints_num', type=int, default=16, help='input of motion')
    # parser.add_argument('--lr', type=float, required=True, default=3e-5, help='input the learning rate')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--negative_margin', type=float, default=0.01)

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--log_dir', default='/home/cjm/CALM/Anyskill/log/')
    parser.add_argument('--model_dir', default='/home/cjm/CALM/Anyskill/output/')
    parser.add_argument('--eval_dir', default='/home/cjm/CALM/Anyskill/eval/')

    parser.add_argument('--dim_movement_enc_hidden', type=int, default=512, help='Dimension of hidden unit in GRU')
    parser.add_argument('--dim_movement_latent', type=int, default=512, help='Dimension of hidden unit in GRU')
    parser.add_argument('--dim_motion_hidden', type=int, default=1024, help='Dimension of hidden unit in GRU')
    parser.add_argument('--dim_coemb_hidden', type=int, default=512, help='Dimension of hidden unit in GRU')

    parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
    parser.add_argument('--save_every_e', type=int, default=50, help='Frequency of printing training progress')
    parser.add_argument('--eval_every_e', type=int, default=30, help='Frequency of printing training progress')
    parser.add_argument('--save_latest', type=int, default=50, help='Frequency of printing training progress')

    opt = parser.parse_args()

    return opt