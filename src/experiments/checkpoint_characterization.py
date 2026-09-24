from src.experiments.get_all_activations import get_all_activations
from src.experiments.erf.netStats import netStats
from src.experiments.erf.netStats_viz import netStats_viz
import torch

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260215_150432-apv3qmwe/files/model/best_model_full.pth"
# model_name = "apv3qmwe" # el tristemente celebre r0 que me dio artifacts en ori

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260217_112708-88dtwmfg/files/model/best_model_full.pth"
# model_name = "88dtwmfg" # r13

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260217_133315-5qny90co/files/model/best_model_full.pth"
# model_name = "5qny90co"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/atto_lc_mini-debugged-three_seeds/WS-S/wandb/offline-run-20251205_101339-ekuyzlow/files/model/best_model_full.pth"
# model_name = "ekuyzlow"

"""
# feb26
base = "/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-works/WS-S/wandb"
end = "/files/model/best_model_full.pth"
runs = [
    "offline-run-20260225_195232-juh0fjor",
    "offline-run-20260225_201558-zrrp6vp4",
    "offline-run-20260225_203710-d9tia7c5",
    "offline-run-20260225_204947-6jxs5hum",
    "offline-run-20260225_213848-57zlnjnu",
    "offline-run-20260225_213759-ptdjc0lw",
    "offline-run-20260225_213944-hw1s7bpk",
    "offline-run-20260225_214110-di5kz1g1"
]

paths = [base + "/" + run + end for run in runs]

nine = "/home/tomasdu/local/WS-S/wandb/run-20260225_131418-18rh9rpd/files/model/best_model_full.pth"
paths.append(nine)

model_names = [run.split("-")[-1] for run in runs]
model_names.append("18rh9rpd")
model_path = paths[0]
model_name = model_names[0]
print(paths)
"""

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260308_135911-tmax5ra3/files/model/best_model_full.pth"
# model_name = "tmax5ra3" # conv on full centered imagenet

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260305_182415-os9zi3ym/files/model/best_model_full.pth"
# model_name = "os9zi3ym" # conv on 10 classes of da costa

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260317_140823-uqvubb6r/files/model/best_model_full.pth"
# model_name = "uqvubb6r" # lc on 10 classes of da costa scratch

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260313_140845-bbsedv10/files/model/best_model_full.pth"
# model_name = "bbsedv10" # cnn on 10 classes of da costa scratch

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260317_140752-w7gd82fi/files/model/best_model_full.pth"
# model_name = "w7gd82fi" # cnn on 10 classes of da costa scratch ("one last hail mary")

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260318_085557-mqi6f48u/files/model/best_model_full.pth"
# model_name = "mqi6f48u" # lc  after bbsedv10

# model_path = "/home/tomasdu/local/fisheye_dws_lindsey/run-20260324_193831-0f1wp6fx/files/model/best_model_full.pth"
# model_name = "0f1wp6fx" # conv with lindsey bottleneck

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws_lc_scot_bottleneck/WS-S/wandb/offline-run-20260325_163919-o3f8g7f7/files/model/best_model_full.pth"
# model_name = "o3f8g7f7" # cnn with lindsey fisheye 4

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws_lc_scot_bottleneck/WS-S/wandb/offline-run-20260325_164318-3prcuzw8/files/model/best_model_full.pth"
# model_name = "3prcruzw8"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260327_182632-8mgw5ddt/files/model/best_model_full.pth"
# model_name = "8mgw5ddt"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260327_183600-4i2ygs9p/files/model/best_model_full.pth"
# model_name = "4i2ugs9p"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260328_185118-wpmy4wt0/files/model/best_model_full.pth"
# model_name = "wpmy4wt0" # r25

# model_path="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260408_200533-7j49e6tu/files/model/best_nonoverfit_model.pth"
# model_name = "7j49e6tu"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260411_182731-87g01is7/files/model/best_nonoverfit_model.pth" # alpha 1
# model_name = "87g01is7"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260411_182736-f86mysvz/files/model/best_nonoverfit_model.pth" # alpha 10
# model_name = "f86mysvz"

# the ones below had all kernels the same
# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260413_172724-5tpg0pd9/files/model/best_nonoverfit_model.pth" # alpha 10
# model_name = "5tpg0pd9"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260413_172738-xd1ycwcn/files/model/best_nonoverfit_model.pth"
# model_name = "xd1ycwcn"

# trying to correct the two from above by using the spatial loss through C as well (as above) but smaller alpha and different init (that's the new thing)
# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260414_120605-6a7ccxhi/files/model/best_nonoverfit_model.pth"
# model_name = "6a7ccxhi" # alpha1

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260414_120654-m6uhoae1/files/model/best_nonoverfit_model.pth"
# model_name = "m6uhoae1"

# toroidal
# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260414_182526-boli0fl9/files/model/best_nonoverfit_model.pth" # alpha10
# model_name = "boli0fl9" # alpha10

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260414_193209-qk4t836t/files/model/best_nonoverfit_model.pth"
# model_name = "qk4t836t"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260415_205854-tui7r52y/files/model/best_nonoverfit_model.pth"
# model_name = "tui7r52y"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260424_152056-6z5km0tn/files/model/best_nonoverfit_model.pth"
# model_name = "6z5km0tn"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260426_160844-4vdz50av/files/model/best_nonoverfit_model.pth"
# model_name = "4vdz50av"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260427_114006-ih3m1gn9/files/model/best_nonoverfit_model.pth"
# model_name = "ih3m1gn9" # r0 with optimizer state

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260428_090729-5ey6v9v5/files/model/epoch_0050.pth"
# model_name = "5ey6v9v5" # r25

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260428_090729-8ecrkrbu/files/model/epoch_0050.pth"
# model_name = "8ecrkrbu"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260428_161951-dj3qy90e/files/model/best_nonoverfit_model.pth"
# model_name = "dj3qy90e" # alpha100 r25 saving the optim

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260428_161930-ohr8lb0v/files/model/best_nonoverfit_model.pth"
# model_name = "ohr8lb0v" # alpha100 control

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260414_193209-qk4t836t/files/model/best_model_full.pth"
# model_name = "qk4t836"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260429_183203-8xuo9bsu/files/model/best_nonoverfit_model.pth"
# model_name = "8xuo9bsu"

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260501_122027-auff9siw/files/model/best_nonoverfit_model.pth"
# model_name = "auff9siw"
# activations_nickname = "alpha100-r0_2part_loss_no_randaug"
# PLOTS_NICKNAME = f"{model_name}-{activations_nickname}" # maybe i do diff plots with same acts.npy files


# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260504_103153-b8ru3qrv/files/model/best_nonoverfit_model.pth"
# model_name = "b8ru3qrv"
# activations_nickname = "alpha100-horiz_conn1"
# PLOTS_NICKNAME = f"{model_name}-{activations_nickname}" # maybe i do diff plots with same acts.npy files
# architecture = 'cornet_dws_lc_hor'

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260527_182921-1f3st7wx/files/model/last_model_full.pth"
# model_name = "1f3st7wx"
# activations_nickname = "alpha100_lc_from_9q7"
# PLOTS_NICKNAME = f"{model_name}-{activations_nickname}" # maybe i do diff plots with same acts.npy files
# architecture = 'cornet_dws_lc'


#model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260528_000347-ww7iyi4w/files/model/best_model_full.pth"
#model_name = "ww7iyi4w"
#activations_nickname = "alpha0_lc_from_9q7"
#PLOTS_NICKNAME = f"{model_name}-{activations_nickname}" # maybe i do diff plots with same acts.npy files
#architecture = 'cornet_dws_lc'

#model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260528_122747-wcvg7vn9/files/model/best_model_full.pth"
#model_name = "wcvg7vn9"
#activations_nickname = "alpha100_lc_from_1f3s"
#PLOTS_NICKNAME = f"{model_name}-{activations_nickname}" # maybe i do diff plots with same acts.npy files
#architecture = 'cornet_dws_lc'

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260528_220652-8ezxq5m0/files/model/best_nonoverfit_model.pth"
# model_name = "8ezxq5m0"
# activations_nickname = "alpha0_lc_from_ww7i"
# PLOTS_NICKNAME = f"{model_name}-{activations_nickname}" # maybe i do diff plots with same acts.npy files
# architecture = 'cornet_dws_lc'

#model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260529_190229-yjzvq14q/files/model/best_nonoverfit_model.pth"
#model_name = "yjzvq14q"
#activations_nickname = "hc1"
#PLOTS_NICKNAME = f"{model_name}-{activations_nickname}" # maybe i do diff plots with same acts.npy files
#architecture = 'cornet_dws_lc_hor'

#model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260602_164304-lgxnvqft/files/model/epoch_0015.pth"
#model_name = "lgxnvqft"
#activations_nickname = "hc_1st_layer_then_lc"
#PLOTS_NICKNAME = f"{model_name}-{activations_nickname}" # maybe i do diff plots with same acts.npy files
#architecture = 'cornet_dws_lc_hor'

#model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260603_214054-fzyiv4gl/files/model/best_nonoverfit_model.pth"
#model_name = "fzyiv4gl"
#activations_nickname = "hc_1st_layer_then_lc"
#PLOTS_NICKNAME = f"{model_name}-{activations_nickname}" # maybe i do diff plots with same acts.npy files
#architecture = 'dws_mix'

#model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260603_224708-ouw910cq/files/model/epoch_0012.pth"
#model_name = "ouw910cq"
#activations_nickname = "conv_hc_then_conv"
#PLOTS_NICKNAME = f"{model_name}-{activations_nickname}" # maybe i do diff plots with same acts.npy files
#architecture = 'dws_mix'

#model_name = "ouw910cq"
#activations_nickname = "conv_hc_then_conv"
#PLOTS_NICKNAME = f"{model_name}-{activations_nickname}" # maybe i do diff plots with same acts.npy files
#architecture = 'dws_mix'

# model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260608_141416-by83k626/files/model/epoch_0024.pth"
# model_name = "by83k626"
# activations_nickname = "conv_hc_then_conv_with_bottleneck"
# PLOTS_NICKNAME = f"{model_name}-{activations_nickname}" # maybe i do diff plots with same acts.npy files
# architecture = 'dws_mix'


model_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_201026-ifybppjl/files/model/best_nonoverfit_model.pth"
model_name = "ify"
activations_nickname = "conv"
PLOTS_NICKNAME = f"{model_name}-{activations_nickname}" # maybe i do diff plots with same acts.npy files
architecture = 'dws_mix'


# Extract activations for 1st layer only (guaranteed by doing "if dwconv.0")
# Returns acts.npy saved to grid_search directory
layers = [
    # "stages.0.0.dwconv",
    "stages.0.0.act",
    #"stages.0.0.feat_recurrent",
    # "stages.0.0.pwconv1",
    # "stages.0.0.pwconv2",
    # "stages.1.0.dwconv",
    # "stages.1.0.pwconv1",
    # "stages.1.0.pwconv2"
    ]
device = "cuda" if torch.cuda.is_available() else "cpu"

gratings_dataset = "/home/tomasdu/repos/datasets/gratings-8phases-256-20SFs/val" # input
acts_output_path = f"/home/tomasdu/repos/trained_models/{model_name}-{activations_nickname}/grid_search" # output

get_all_activations(checkpoint=model_path, model_name=model_name, layers=layers, device=device, gratings_dataset=gratings_dataset, acts_output_path=acts_output_path,architecture=architecture)

# For 1st layer only. Receives model_name where the acts.npy per layer are stored;
# since we saved only dwconv.0, it will only process that layer.
# Saves SF pref, ori pref for normal and special channels

manifest_csv = f"{gratings_dataset}/manifest.csv"
# Subset of SF indices (into the manifest freq list) to compute orientation maps for.
# None = all SFs. Example: list(range(0, 50, 5)) gives every 5th SF.
ORI_SF_INDICES = list(range(0, 20, 4))
# "vector_strength": resultant / sum(acts), matches Lu et al. 2025
# "resultant":       raw √(x²+y²), unnormalized
SEL_MODE = "resultant"
# Saturation encoding for the unfolded sheet plot:
# "uniform":      all vivid colours, matches Lu et al. 2025 paper in theory but not in their code
# "selectivity":  desaturated = untuned channels
SAT_MODE = "uniform"
netStats(model_name=model_name, manifest_csv=manifest_csv, acts_output_path=acts_output_path, ori_sf_indices=ORI_SF_INDICES)

# For 1st layer only. Receives model_name as well to look into the files saved by netStats
# Plots SF pref, ori pref and daCosta plot about SF from the gratings
# SPECIAL_CHANNELS = [31,29,20,17,13,11,5,1]
SPECIAL_CHANNELS = None
for layer in layers:
    model_outpath = f"{acts_output_path}/{layer}"
    netStats_viz(model_name=model_name, special_channels=SPECIAL_CHANNELS, layer_name=layer, model_outpath=model_outpath, manifest_csv=manifest_csv, ori_sf_indices=ORI_SF_INDICES, sel_mode=SEL_MODE, sat_mode=SAT_MODE, plots_nickname=PLOTS_NICKNAME)

gradmaps = False

# if gradmaps:
