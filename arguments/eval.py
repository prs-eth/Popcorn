import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)
parser.add_argument('-r', '--resume', nargs='+', help='argument should be name of the model to be trained')
parser.add_argument("-treg", "--target_regions", nargs='+', default=["pri2017"], help="the target domains")
parser.add_argument("-S1", "--Sentinel1", action='store_true', help="")
parser.add_argument("-S2", "--Sentinel2", action='store_true', help="")
parser.add_argument("-NIR", "--NIR", action='store_true', help="")
parser.add_argument("-m", "--model", help='', type=str, default="POPCORN")
parser.add_argument("-occmodel", "--occupancymodel", help='', action='store_true')
parser.add_argument("-sinp", "--segmentationinput", help='', action='store_true')
parser.add_argument("-binp", "--buildinginput", help='', action='store_true')
parser.add_argument("-senbuilds", "--sentinelbuildings", help='', action='store_true')
parser.add_argument("-fs", "--fourseasons", action='store_true', help="")
parser.add_argument('-fe', '--feature_extractor', type=str, help=' ', default="DDA")
parser.add_argument('-pret', '--pretrained', help='', action='store_true')
parser.add_argument('-tlevel', '--train_level', nargs='+', default=["coarse"], help="needs to be know to perform the adjustment") 
parser.add_argument("-binit", "--biasinit", help='', type=float, default=0.75)

# misc
parser.add_argument('--save-dir', default='./results', help='Path to directory where models and logs should be saved')
parser.add_argument('-w', '--num_workers', help='', type=int, default=8)
parser.add_argument("-wp", "--wandb_project", help='', type=str, default="POPCORN")
parser.add_argument("--seed", help='', type=int, default=1610)
parser.add_argument("--in_memory", action='store_true', help='')

args = parser.parse_args()