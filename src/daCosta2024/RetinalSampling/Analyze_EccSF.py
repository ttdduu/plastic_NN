# from tensorflow.python.keras.models import model_from_json
from src.daCosta2024.RetinalSampling.ImSetNorm import Normalize
from scipy.stats import sem
# import tensorflow.python.keras as keras
import pandas as pd
import glob
import cv2
import numpy as np
import os
import re
import math
import scipy
import matplotlib.pyplot as plt
import matplotlib
import torch
from typing import List, Optional
from src.models.convnext_atto_lc_8254a59_RMS_shrunk import CustomConvNeXtAttoLC_8254a59_RMS_shrunk
from src.models.cornet_dwsep import CustomCornetZDWSep
from src.models.cornet_dwsep import CustomCornetDWSep
from src.models.cornet_dws_lc import CustomCornetDWSepLC
from src.models.cornet_z_daCosta import CustomCornetZDaCosta
from src.models.cornet_dwsep_retina import CustomCornetDWSepRetina
from src.models.dws_mix import DWSMix

# LAYER="stages.0.0.pwconv1"
# LAYER="stages.0.0.dwconv"

class AnalyzeEccSf:
    """
    A class for analyzing and visualizing data related to Eccentricity and Spatial Frequency using sinring stimuli.

    This class is designed to handle the analysis and visualization of neural network model outputs
    in the context of eccentricity and spatial frequency, with a focus on sinring stimuli. It includes methods
    to load pre-trained models, process sinring stimulus images, analyze specific model layers,
    and create customized plots to visualize the results.

    Attributes:
        root (str): Root directory path for various data and model folders.
        cnn_model_root (str): Sub-directory for CNN model.
        rsl_model_root (str): Sub-directory for RSL model.
        layer_numbers (dict): Mapping of layer names to their indices in the model's layer list.
        model (keras.Model): The loaded neural network model.
        model_cut (keras.Model): A sub-model cut at a specific layer.
        stim_root (str): Root directory path for stimulus images.
        cnn_stim_folder (str): Sub-directory for CNN stimulus images.
        rsl_stim_folder (str): Sub-directory for RSL stimulus images.
        sub_folders (list): Sub-folders for different eccentricities.
        layers (list): Names of neural network layers.
        outsize (dict): Mapping of layer names to their output sizes.
        rsl_result_folder (str): Result directory for RSL analysis.
        cnn_result_folder (str): Result directory for CNN analysis.
        savetype (str): File format for saving plots.
        frequency (list): List of spatial frequencies.
        cdeg_alt (list): List of circular degrees with alternative labels.
        cdeg (list): List of circular degrees.
        rads (list): List of radii corresponding to eccentricities.
        circ (list): List of circumferences corresponding to eccentricities.
        results (dict): Dictionary to store analysis results.
        results_keys (list): List of keys in the results dictionary.
        results_sem (dict): Dictionary to store analysis results' standard errors.

    Methods:
        __init__() -> None:
            Initialize class attributes with default values.

        load_model(model: str) -> None:
            Load a pre-trained neural network model based on the specified type ("cnn" or "rsl").

        select_model_layer(layer: str) -> None:
            Select a specific layer from the loaded model for further analysis.

        load_stim(path: str) -> numpy.ndarray:
            Load stimulus images from the specified path.

        g1(r, rad, ring) -> float:
            Calculate a Gaussian function value given parameters.

        reshape_matrix(matrix: numpy.ndarray) -> numpy.ndarray:
            Reshape a matrix for visualization.

        save_featuremaps(maps: numpy.ndarray, ecc: str, sf: str) -> None:
            Save feature maps as images for a specific eccentricity and spatial frequency.

        save_image(image: numpy.ndarray, name: str, root: str, ecc: str, sf: str) -> None:
            Save an image with a specific name and location.

        exp_decay_func(x, m, t, b) -> float:
            Exponential decay function for fitting.

        exp_func(x, t, b) -> float:
            Exponential function for fitting.

        f_test_regression(R2: float, p: float, n: int) -> Tuple[float, float]:
            Perform F-test for regression analysis.

        analyze_model(layers: list) -> None:
            Analyze the selected model layers using loaded stimuli.

        create_plot() -> None:
            Create a customized plot to visualize analyzed data.

    Note:
        - This class assumes that the required libraries (e.g., keras, numpy, pandas, matplotlib) are imported in the caller's scope.
        - Class attributes are initialized in the __init__() method.
        - Methods prefixed with @staticmethod are utility methods that don't rely on class instance attributes.
    """
    def __init__(
        self,
        LAYER,
        *,
        stim_root: Optional[str] = None,
        save_featuremaps: bool = False,
    ):
        # ====== modify for each run

        # self.rsl_model_root = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260317_123400-smi9593c/files/model/best_model_full.pth"
        # model_name = "smi9593c" # really nice conv

        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260317_120709-h04iuizq/files/model/best_model_full.pth"
        # model_name = "h04iuizq" # cnn with same dims as the shit lc

        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260317_103554-9ybvy4nq/files/model/best_model_full.pth"
        # model_name = "9ybvy4nq" # really nice lc

        # final
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260317_140752-w7gd82fi/files/model/best_model_full.pth"
        # model_name = "w7gd82fi" # final

        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260317_140823-uqvubb6r/files/model/best_model_full.pth"
        # model_name = "uqvubb6r" # final

        # the old one
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260215_150432-apv3qmwe/files/model/best_model_full.pth"
        # model_name = "apv3qmwe" # old

        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260318_085557-mqi6f48u/files/model/best_model_full.pth"
        # model_name = "mqi6f48u" # lc after bbsedv10

        # self.rsl_model_root="/home/tomasdu/local/fisheye_dws_lindsey/run-20260324_193831-0f1wp6fx/files/model/best_model_full.pth"
        # model_name = "0f1wp6fx" # conv with lindsey bottleneck

        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws_lc_scot_bottleneck/WS-S/wandb/offline-run-20260325_163919-o3f8g7f7/files/model/best_model_full.pth"
        # model_name = "o3f8g7f7" # cnn with lindsey fisheye 4

        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws_lc_scot_bottleneck/WS-S/wandb/offline-run-20260325_164318-3prcuzw8/files/model/best_model_full.pth"
        # model_name = "3prcruzw8" # won't run it for now, i just now it'll be good

        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260327_182632-8mgw5ddt/files/model/best_model_full.pth"
        # model_name = "8mgw5ddt"

        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260327_183600-4i2ygs9p/files/model/best_model_full.pth"
        # model_name = "4i2ygs9p"

        # self.rsl_model_root = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260328_185118-wpmy4wt0/files/model/best_model_full.pth"
        # model_name = "wpmy4wt0" # r25

        # self.rsl_model_root = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260413_172724-5tpg0pd9/files/model/best_nonoverfit_model.pth" # alpha 10
        # model_name = "5tpg0pd9"

        # self.rsl_model_root = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260413_172738-xd1ycwcn/files/model/best_nonoverfit_model.pth"
        # model_name = "xd1ycwcn"

        # self.rsl_model_root = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260414_193209-qk4t836t/files/model/best_nonoverfit_model.pth"
        # model_name = "gk4t836t"

        # self.rsl_model_root = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260527_182921-1f3st7wx/files/model/last_model_full.pth"
        # model_name = "1f3st7wx" # lc alpha 100 last epoch

        # self.rsl_model_root = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260528_000347-ww7iyi4w/files/model/last_model_full.pth"
        # model_name = "ww7iyi4w" # lc alpha 0 last epoch

        # self.rsl_model_root = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260603_214054-fzyiv4gl/files/model/best_nonoverfit_model.pth"
        # model_name = "fzyiv4gl" # recurrent lc first layer then only lc

        #self.rsl_model_root = "/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260603_224708-ouw910cq/files/model/epoch_0012.pth"
        #model_name = "ouw910cq" # conv hcs then conv hcs

        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260608_141416-by83k626/files/model/epoch_0024.pth"
        # model_name= "by83k626" # conv-hc + conv_bottleneck on 10 classes above 70%val

        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260726_164054-lf5a8dxn/files/model/best_nonoverfit_model.pth"
        # model_name='lf5a8'

        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_122727-vi8zcmop/files/model/best_nonoverfit_model.pth"
        # model_name = 'vi8z'

        # self.rsl_model_root='/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260728_201026-ifybppjl/files/model/best_nonoverfit_model.pth'
        # model_name='ify'

        self.rsl_model_root='/home/tomasdu/repos/experiments/plastic_NNs/active/WS-S/wandb/offline-run-20260729_154917-aw41x6hk/files/model/best_nonoverfit_model.pth'
        model_name='aw41'

        # self.lc_weights = None
        # self.conv_weights = None
        # self.weights_path = None
        # self.lc_weights = self.rsl_model_root
        # self.conv_weights = self.rsl_model_root
        self.weights_path = self.rsl_model_root

        nickname ="stemKS3" # change for each run of the script (e.g. same exact model but diff epoch)

        #self.Model = CustomCornetDWSepLC
        self.Model = DWSMix
        # self.Model = CustomConvNeXtAttoLC_8254a59_RMS_shrunk
        # self.Model = CustomCornetDWSep
        # self.Model = CustomCornetDWSepRetina
        # self.Model = CustomCornetZDWSep
        # self.Model = CustomCornetZDaCosta


        # model_path = self.cnn_model_root
        self.model_path = self.rsl_model_root
        # ======


        output_nickname = f"{model_name}_{nickname}"
        # rootname="fisheye_sinrings_rfov30_more_granular" # this is the good dataset now, leave it as is
        rootname="fisheye_sinrings_rfov30_more_granular"
        self.root = f"/home/tomasdu/repos/datasets/{rootname}/"
        self.layer_numbers = {LAYER: 0} # there's other layers but i don't care about them for now, is it ok if i leave this dict as is, or will it break the code?
        self.model = None
        self.model_cut = None


        # If you generated an expanded SF set with create_RSL_more_SFs.py, point this to
        # self.stim_root = "/home/tomasdu/repos/datasets/daCosta2024-remake/stimuli/sinrings/"
        self.stim_root = stim_root or f"/home/tomasdu/repos/datasets/{rootname}/stimuli/sinrings/"
        self.cnn_stim_folder = "undistorted/"
        self.rsl_stim_folder = "distorted/"
        # Will be discovered from disk at runtime (per stim_folder).
        self.sub_folders = ["1_Ecc/", "2.8_Ecc/", "4.7_Ecc/", "6.6_Ecc/", "8.5_Ecc/"]
        self.layers = [LAYER]
        self.outsize = {LAYER: 256}
        self.rsl_result_folder = f"/home/tomasdu/repos/results/eccSf/{rootname}/{output_nickname}/{LAYER}/rsl/"
        self.cnn_result_folder = f"/home/tomasdu/repos/results/eccSf/{rootname}/{LAYER}/cnn/"
        self.savetype = 'png'
        self.save_featuremaps_enabled = bool(save_featuremaps)

        # Legacy (paper) SF labels. Will be replaced by whatever is found on disk.
        self.frequency = [4, 10, 16, 22, 28, 34, 40, 46]  # plotted x positions (label = 2*n)
        self.cdeg = ['2/π e', '5/π e', '8/π e', '11/π e', '14/π e', '17/π e', '20/π e', '23/π e']

        self.rads = [1, 2.8, 4.7, 6.6, 8.5]  # degrees (will be replaced by discovered ecc folders)
        self.circ = [(2 * np.pi * r) for r in self.rads]

        self.results = {}
        self.results_keys = []
        self.results_sem = {}
        self.sf_labels = None  # sorted list of label_2n values present in the dataset
        self.sf_ticklabels = None  # strings like 'n/π e' (n inferred from label/2)

    def load_model(self, model):
        """
                Load a pre-trained neural network model.

                Parameters:
                    model (str): The model type to load ("cnn" for CNN model, "rsl" for RSL model).

                Returns:
                    None
                """
        import torch
        from types import SimpleNamespace
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = SimpleNamespace()
        config.num_classes = 50
        config.fisheye_apply = True
        config.fisheye_C = 1
        config.fisheye_K = -7
        config.fisheye_rfov = 30
        config.no_stem=False
        # if self.Model == DWSMix:
            # config.lateral_target = "dwconv_out"
            # config.recurrent_norm_mode  = "rms"      # or whatever you trained with
            # config.lateral_init_mode    = "kaiming_positive_shift"
            # #config.lateral_gain_init    = 1.5
            # config.lateral_kernel_size  = 5
            # config.recurrent_timesteps  = 1   # makes [Init] match your override
        self.model = self.Model(
            config=config,
            lc_weights = self.rsl_model_root,
            # conv_weights = self.rsl_model_root,
            # weights_path = self.rsl_model_root,
        )
        self.model.eval()
        self.model.to(self.device)
        print(f"\nModel loaded on {self.device}")

        self.stim_folder = self.rsl_stim_folder
        self.model_folder = self.rsl_model_root # useless
        self.result_folder = self.rsl_result_folder

    @staticmethod
    def _parse_sinring_filename(path: str):
        """
        Expected filename format from OSF and create_RSL_more_SFs.py:
            {idx}_{label}_{ecc}.jpg
        where label = 2*n (integer) and ecc is '1', '2.8', etc.
        """
        base = os.path.basename(path)
        # NOTE: raw string => use single backslashes for regex escapes.
        m = re.match(r"^(?P<idx>\d+)_(?P<label>\d+)_(?P<ecc>[0-9]+(?:\.[0-9]+)?)\.jpg$", base)
        if not m:
            return None
        return int(m.group("idx")), int(m.group("label")), float(m.group("ecc"))

    def _discover_ecc_folders(self) -> List[str]:
        root = os.path.join(self.stim_root, self.stim_folder)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Stimulus folder not found: {root}")
        ecc_dirs = []
        for d in os.listdir(root):
            p = os.path.join(root, d)
            if not os.path.isdir(p):
                continue
            if not d.endswith("_Ecc"):
                continue
            try:
                ecc = float(d.replace("_Ecc", ""))
            except Exception:
                continue
            ecc_dirs.append((ecc, d + "/"))
        ecc_dirs.sort(key=lambda t: t[0])
        self.rads = [e for e, _ in ecc_dirs]
        self.circ = [(2 * np.pi * r) for r in self.rads]
        return [d for _, d in ecc_dirs]

    def forward_to_layer(self, x, layer_name):
        print(layer_name)

        self._features = None

        with torch.no_grad():
            _ = self.model(x)

        if self._features is None:
            raise RuntimeError("Hook did not capture any features")

        return self._features



    def select_model_layer(self, layer):
        """
            Select a specific layer from the loaded model.

            Parameters:
                layer (str): The layer name to select (e.g., "V1", "V2", "V4").

            Returns:
                None
        """
        layer_n = self.layer_numbers[layer]
        print(f"Cutting CNN model at layer {layer}")

        self._features = None
        def hook(module, input, output):
            self._features = output

        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()

        module_dict = dict(self.model.named_modules())
        print(module_dict)
        target_module = module_dict[layer]
        self._hook_handle = target_module.register_forward_hook(hook)

    @staticmethod
    def load_stim(path):
        """
            Load stimulus images from a given path.

            Parameters:
                path (str): The path to the folder containing stimulus images.

            Returns:
                numpy.ndarray: An array containing loaded stimulus images.
        """
        filepath = path + "*.jpg"
        files = glob.glob(filepath)
        parsed = []
        for f in files:
            meta = AnalyzeEccSf._parse_sinring_filename(f)
            if meta is None:
                continue
            idx, label, ecc = meta
            img = cv2.imread(f, 1)
            parsed.append({"path": f, "idx": idx, "label": label, "ecc": ecc, "img": img})
        parsed.sort(key=lambda r: (r["label"], r["idx"]))
        return parsed

    #@staticmethod
    #def g1(r, rad, ring):
    #    return np.exp(-(np.square(r - rad) / (2 * np.square(ring))))

    def reshape_matrix(self, matrix):
        """
            Reshape a multi-dimensional matrix for visualization.

            This method reshapes a multi-dimensional input matrix into a two-dimensional array, where each column
            represents a single channel or feature. This reshaping is primarily used for visualization purposes.

            Parameters:
                matrix (numpy.ndarray): The input multi-dimensional matrix to be reshaped.

            Returns:
                numpy.ndarray: A two-dimensional array obtained by reshaping the input matrix.
        """
        # output = np.asarray([matrix[0, :, :, i] for i in range(matrix.shape[3])])
        return matrix[0]

    def save_featuremaps(self, maps, ecc, sf):
        """
            Save feature maps as images for a specific eccentricity and spatial frequency.

            This method takes a set of feature maps and saves them as image files for a particular eccentricity and
            spatial frequency combination. The images are saved in a designated result folder.

            Parameters:
                maps (numpy.ndarray): Feature maps to be saved as images.
                ecc (str): Eccentricity label for the saved images.
                sf (str): Spatial frequency label for the saved images.

            Returns:
                None
        """
        # selection = [0, 1, 2, 5]
        maps = self.reshape_matrix(maps)
        os.makedirs(self.result_folder + ecc + sf, exist_ok=True)
        # for i, e in enumerate(selection):
        m = maps[0, 2:-2, 2:-2]
        plt.imsave(self.result_folder + ecc + sf + "/{}.tiff".format(0), m)

    @staticmethod
    def save_image(image, name, root, ecc, sf):
        plt.imsave(root + name + "/" + ecc + sf + "/population.svg", image)

    @staticmethod
    def exp_decay_func(x, m, t, b):
        return m * np.exp(-t * x) + b

    @staticmethod
    def exp_func(x, t, b):
        return x ** t + b

    @staticmethod
    def f_test_regression(R2, p, n):
        dfn = p - 1
        dfd = n - dfn - 1
        numerator = R2 / dfn
        denominator = (1 - R2) / dfd
        f = numerator / denominator
        p = 1 - scipy.stats.f.cdf(f, dfn, dfd)  # find p-value of F test statistic
        return f, p

    def fontsizes(self, fontsize):  # region Fonts
        matplotlib.rcParams['font.family'] = 'sans-serif'
        plt.rc('font', family='arial')
        plt.rc('font', size=fontsize)  # controls default text size
        plt.rc('axes', titlesize=fontsize)  # fontsize of the title
        plt.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=fontsize)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=fontsize)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=fontsize)  # fontsize of the legend

    def analyze_model(self, layers):
        """
        Analyze the selected model layers using loaded sinring stimuli.

        This method performs analysis on the neural network model using the loaded sinring stimuli images.
        Prior to running this function, ensure that you have preloaded a model using the 'load_model()' method.

        For each specified layer, the method calculates the average response and standard error of the response
        across multiple loaded sinring stimuli images at different eccentricities. The analysis results are stored
        in the 'results' and 'results_sem' dictionaries of the class instance.

        Parameters:
            layers (list): A list of layer names to be analyzed.

        Returns:
            None

        Note:
            - Prior to running this function, you should use the 'load_model()' method to preload a neural network model.
            - The term "loaded sinring stimuli" refers to the set of stimulus images containing sinusoidal gratings
              arranged in ring-like patterns.
            - This method relies on the 'select_model_layer' method to choose the appropriate layer for analysis.
            - The analysis results are stored in the 'results' and 'results_sem' dictionaries,
              where each entry corresponds to a specific eccentricity.
        """
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        self.values = []
        self.sem_values = []
        for layer in layers:
            self.select_model_layer(layer)

            # Discover eccentricity folders for the currently selected stimulus set (undistorted/ or distorted/)
            self.sub_folders = self._discover_ecc_folders()
            # Initialize results dicts keyed by eccentricity string ("1", "2.8", ...)
            self.results = {f"{e:g}": [] for e in self.rads}
            self.results_sem = {f"{e:g}": [] for e in self.rads}
            self.results_keys = [f"{e:g}" for e in self.rads]

            for j, folder in enumerate(self.sub_folders):
                prediction = []
                sems = []
                path = self.stim_root + self.stim_folder + folder
                stim = self.load_stim(path)
                print(path)
                if not stim:
                    raise RuntimeError(f"No sinring images found in: {path}")

                # Determine SF labels / ticklabels from the first eccentricity folder.
                # Assumes the same set of labels exists for all eccentricities.
                if self.sf_labels is None:
                    self.sf_labels = [int(r["label"]) for r in stim]
                    self.sf_ticklabels = [f"{int(l//2)}/π e" for l in self.sf_labels]
                    self.frequency = list(self.sf_labels)
                    self.cdeg = list(self.sf_ticklabels)

                # Resize images from 256x256 to 144x144
                # resized_stim = []
                # for img in stim:
                    # resized_img = cv2.resize(img, (144, 144), interpolation=cv2.INTER_LINEAR)
                    # resized_stim.append(resized_img)
                # stim = np.array(resized_stim)

                raw_imgs = np.stack([row["img"] for row in stim]).astype(np.float32)
                # normed_imgs = Normalize(raw_imgs, 1)  # global min-max → [0, 1]
                normed_imgs = raw_imgs
                for i, row in enumerate(stim):
                    row["img"] = normed_imgs[i]

                for k, row in enumerate(stim):
                    img = row["img"]
                    label = int(row["label"])
                    n = label / 2.0

                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                    # img already in [0, 1] from Normalize — no /255 needed
                    img = img / 255
                    img = (img-mean) / std
                    img = img.to(self.device)
                    print("aaaaaaa")
                    print(img.shape)
                    with torch.no_grad():
                        feature_maps = self.forward_to_layer( img,layer)
                    print(feature_maps.shape)
                    feature_maps = feature_maps.cpu().numpy()
                    if self.save_featuremaps_enabled:
                        sf_tag = f"{n:g}_over_pi_e"
                        self.save_featuremaps(feature_maps, folder, sf_tag)
                    activity = np.mean(feature_maps)
                    sem_l = sem(feature_maps, axis=None)
                    prediction.append(activity)
                    sems.append(sem_l)

                print(prediction)

                self.results[self.results_keys[j]] = np.divide(prediction, np.max(prediction))
                self.results_sem[self.results_keys[j]] = np.divide(sems, np.max(prediction))
                self.values.append(np.divide(prediction, np.max(prediction)))
                self.sem_values.append(np.divide(sems, np.max(prediction)))

    def create_plot(self,layer):
        """
            Create a customized plot visualizing normalized layer responses.

            This method generates a customized plot to visualize the normalized layer responses based on the loaded
            sinring stimuli and the analyzed model outputs. The plot includes line plots representing different
            eccentricities and shaded regions indicating the standard error of the response.

            The x-axis of the plot corresponds to spatial frequency (in cycles per ring), while the y-axis represents
            the normalized layer response. The plot is designed to provide insights into how the neural network model's
            responses vary with different eccentricities and spatial frequencies.

            Parameters:
                None

            Returns:
                None

            Note:
                - Prior to running this function, ensure that you have loaded sinring stimuli and performed model analysis
                    using the 'load_stim' and 'analyze_model' methods.
                - The plot is customized with appropriate labels, colors, and styles to enhance readability.
                - The analysis results are used to generate the plot, including normalized responses and standard errors.
        """
        sf1_color = '#91C4FF'
        sf2_color = '#FFBF4A'
        sf3_color = '#76CC8C'
        sf4_color = '#FF7C7C'
        sf5_color = '#BCA3FF'

        line1_c = "#2c8eff"
        line2_c = "#ffa500"
        line3_c = "#34d15c"
        line4_c = "#ff3131"
        line5_c = "#7642ff"

        width = 1.0
        df = pd.DataFrame(self.results)
        sem = pd.DataFrame(self.results_sem)
        ms = 5

        self.fontsizes(7)
        fig1, ax = plt.subplots(figsize=(3.4, 2.4))
        matplotlib.rcParams['font.family'] = 'sans-serif'
        plt.rc('font', family='arial')
        ax.plot(self.frequency, self.values[0], marker=None, ms=ms, ls="-", linewidth=width, color=line1_c,
                label="1\N{DEGREE SIGN}")
        ax.fill_between(self.frequency, self.values[0] - self.sem_values[0] * 1.96, self.values[0] + self.sem_values[0]
                        * 1.96, color=sf1_color, alpha=0.45)
        ax.plot(self.frequency, self.values[1], marker=None, ms=ms, ls="-", linewidth=width, color=line2_c,
                label="2.8\N{DEGREE SIGN}")
        ax.fill_between(self.frequency, self.values[1] - self.sem_values[1] * 1.96, self.values[1] + self.sem_values[1]
                        * 1.96, color=sf2_color, alpha=0.45)
        ax.plot(self.frequency, self.values[2], marker=None, ms=ms, ls="-", linewidth=width, color=line3_c,
                label="4.7\N{DEGREE SIGN}")
        ax.fill_between(self.frequency, self.values[2] - self.sem_values[2] * 1.96, self.values[2] + self.sem_values[2]
                        * 1.96, color=sf3_color, alpha=0.45)
        ax.plot(self.frequency, self.values[3], marker=None, ms=ms, ls="-", linewidth=width, color=line4_c,
                label="6.6\N{DEGREE SIGN}")
        ax.fill_between(self.frequency, self.values[3] - self.sem_values[3] * 1.96, self.values[3] + self.sem_values[3]
                        * 1.96, color=sf4_color, alpha=0.45)
        ax.plot(self.frequency, self.values[4], marker=None, ms=ms, ls="-", linewidth=width, color=line5_c,
                label="8.5\N{DEGREE SIGN}")
        ax.fill_between(self.frequency, self.values[4] - self.sem_values[4] * 1.96, self.values[4] + self.sem_values[4]
                        * 1.96, color=sf5_color, alpha=0.45)
        x_labl = "Spatial Frequency (cpr)"
        plt.xlabel(x_labl, fontsize=7, family='sans-serif', labelpad=-2.5)
        plt.ylabel("Normalized Layer Response", fontsize=7, family='sans-serif', labelpad=-4)
        n_sf = len(self.frequency)
        if n_sf <= 10:
            ax.set_xticks(self.frequency)
            ax.set_xticklabels(self.cdeg, fontsize=5, rotation=45, ha='right')
        else:
            step = max(1, n_sf // 8)
            tick_idx = list(range(0, n_sf, step))
            if (n_sf - 1) not in tick_idx:
                tick_idx.append(n_sf - 1)
            ax.set_xticks([self.frequency[i] for i in tick_idx])
            ax.set_xticklabels([self.cdeg[i] for i in tick_idx], fontsize=5, rotation=45, ha='right')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.minorticks_on()
        ax.tick_params(which='major', labelsize=5)
        ax.tick_params('both', length=0, width=0, which='minor', labelsize=5)
        # plt.ylim(0.2, 1.3)
        # ax.set_yticklabels(('0.7', '', '', '1.0'))
        plt.locator_params(axis="y", nbins=5)
        # Legacy code hardcoded a subset of tick labels for exactly 8 SFs.
        # For expanded SF sets, we keep the full dynamically derived ticklabels above.
        ax.legend(title=None, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0), prop={'size': 7})
        plt.tight_layout()
        os.makedirs(self.result_folder, exist_ok=True)
        plt.savefig(self.result_folder + f'Normalized_Ecc_{layer}' + '.' + self.savetype, dpi=600, format=self.savetype)
        # plt.show()

    def fit_model(self):
        """
            Fit an exponential function to model preferences for eccentricity using curve fitting.

            This method calculates the best-fit exponential function to model preferences for eccentricity based on the
            normalized responses obtained from the analysis of the neural network model. The fitting is performed using
            the `scipy.optimize.curve_fit` function.

            The method first identifies the preferred spatial frequency for each eccentricity by finding the frequency
            with the highest normalized response. It then fits an exponential decay function to these preferred frequencies
            as a function of eccentricity.

            The fitted parameters, such as slope (m), time constant (t), and base level (b), are printed, along with
            the coefficient of determination (R²) and results of an F-test to assess the goodness of fit.

            Finally, a customized plot is generated to visualize the fitted exponential function alongside the actual
            model preferences. The plot includes appropriate labels, colors, and styles to enhance readability.

            Parameters:
                None

            Returns:
                None

            Note:
                - This method requires that the analysis results have been obtained through the 'analyze_model' method.
                - The fitting is performed using the 'exp_decay_func' defined in this class.
                - The fitted parameters and statistical values are printed for reference.
                - The plot generated showcases the fitted exponential function and model preferences.
            """
        # Preferred SF per eccentricity.
        # We convert from dataset label (2*n) to numeric cycles-per-radian (cpr):
        #   cpr = (n/pi) * e_deg  where e_deg is the eccentricity of the ring.
        model_pref = []
        for j, e_deg in enumerate(self.rads):
            ind = self.results_keys[j]
            l = int(np.argmax(self.results[ind]))
            label_2n = float(self.frequency[l])
            n = label_2n / 2.0
            cpr = (n / math.pi) * float(e_deg)
            model_pref.append(cpr)

        print(f"  model_pref (cpr) = {model_pref}")
        print(f"  rads             = {self.rads}")

        # Fit exponential decay.  Use data-driven initial guesses and allow
        # more iterations so the fit converges for arbitrary SF ranges.
        p0 = (max(model_pref) - min(model_pref), 0.3, min(model_pref))
        fit_ok = True
        try:
            params, cv = scipy.optimize.curve_fit(
                self.exp_decay_func, self.rads, model_pref, p0, maxfev=10000,
            )
        except RuntimeError as e:
            print(f"  [WARNING] curve_fit failed: {e}")
            print("  Plotting data without fitted curve.")
            fit_ok = False
            params = (0, 0, 0)

        m_Freq, t_Freq, b_Freq = params

        if fit_ok:
            squaredDiffs = np.square(np.array(model_pref) - self.exp_decay_func(np.array(self.rads), m_Freq, t_Freq, b_Freq))
            squaredDiffsFromMean = np.square(np.array(model_pref) - np.mean(np.array(model_pref)))
            denom = np.sum(squaredDiffsFromMean)
            rSquared_exp_V1_Freq = 1 - np.sum(squaredDiffs) / denom if denom > 0 else float('nan')
            sample_size = len(model_pref)
            f_exp_V1_Freq, p_exp_V1_Freq = self.f_test_regression(rSquared_exp_V1_Freq, 3, sample_size)
            print(f"m={m_Freq}, t={t_Freq}, b={b_Freq}, n={sample_size}")
            print(f"R² for V1 RF exponential fitting = {rSquared_exp_V1_Freq} F = {f_exp_V1_Freq}, p = {p_exp_V1_Freq}")

        self.fontsizes(7)
        fig1, ax1 = plt.subplots(figsize=(2.8, 2.025))
        matplotlib.rcParams['font.family'] = 'sans-serif'
        plt.rc('font', family='arial')
        ax1.plot(self.rads, model_pref, 'o-', linewidth=1.5, color='grey', label="Model preference")
        if fit_ok:
            plt.plot(self.rads, self.exp_decay_func(np.array(self.rads), m_Freq, t_Freq, b_Freq), '--',
                                    label="Fitted exponential function")
        ax1.legend(framealpha=1, markerscale=1, loc='upper right', frameon=False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        # y-axis is numeric cpr preference now
        ax1.set_xlim([min(self.rads), max(self.rads)])
        plt.locator_params(axis="x", nbins=9)
        ax1.set_xticklabels((f"{self.rads[0]:g}", "", "", "", "", "", "", "", f"{self.rads[-1]:g}"), size=7)
        ax1.minorticks_off()
        ax1.tick_params('both', length=00, width=0, which='minor', labelsize=7, size=7)
        plt.ylabel("cpr Preference", labelpad=-12, family='arial', fontsize=7)  # -7
        plt.xlabel("Eccentricity (degree)", labelpad=-2, family='arial', fontsize=7)
        plt.tight_layout()
        os.makedirs(self.result_folder, exist_ok=True)
        plt.savefig(self.result_folder + 'Ecc_degree' + '.' + self.savetype, dpi=600, format=self.savetype)
        # plt.show()



        # self.rsl_model_root = "/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260215_150432-apv3qmwe/files/model/best_model_full.pth"
        # self.rsl_model_root = "/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-ILSVRC_centered-new_dwSep/WS-S/wandb/offline-run-20260224_114827-vk6tty1r/files/model/best_model_full.pth"
        # self.rsl_model_root = "/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-ILSVRC_centered-new_dwSep/WS-S/wandb/offline-run-20260224_143552-1kmgtyau/files/model/best_model_full.pth"
        # self.rsl_model_root = "/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-ILSVRC_centered-new_dwSep/WS-S/wandb/offline-run-20260224_145159-lwmmyydf/files/model/best_model_full.pth"
        # self.rsl_model_root = "/home/tomasdu/local/WS-S/wandb/run-20260224_195454-na2alnms/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/local/WS-S/wandb/run-20260224_195454-na2alnms/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/local/WS-S/wandb/run-20260225_104730-qerhwlej/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/local/WS-S/wandb/run-20260225_120653-mip4otoe/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/local/WS-S/wandb/run-20260225_131418-18rh9rpd/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-works/WS-S/wandb/offline-run-20260225_162216-yqynomo8/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-works/WS-S/wandb/offline-run-20260225_170213-b7j240f0/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-works/WS-S/wandb/offline-run-20260225_172339-21oisq1i/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-works/WS-S/wandb/offline-run-20260225_180402-912jrg6l/files/model/best_model_full.pth"
        # redoing --------------------------------
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-works/WS-S/wandb/offline-run-20260225_195232-juh0fjor/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-works/WS-S/wandb/offline-run-20260225_201558-zrrp6vp4/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-works/WS-S/wandb/offline-run-20260225_203710-d9tia7c5/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-works/WS-S/wandb/offline-run-20260225_204947-6jxs5hum/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-works/WS-S/wandb/offline-run-20260225_213759-ptdjc0lw/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-works/WS-S/wandb/offline-run-20260225_213848-57zlnjnu/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-works/WS-S/wandb/offline-run-20260225_213944-hw1s7bpk/files/model/best_model_full.pth"
        # self.rsl_model_root="/home/tomasdu/repos/experiments/plastic_NNs/active/RSL-works/WS-S/wandb/offline-run-20260225_214110-di5kz1g1/files/model/best_model_full.pth"

        # weights_path=model_path
        # weights_path = "/home/tomasdu/repos/plastic_NNs/src/daCosta2024/RetinalSampling/RSL_model/model.h5"
        # weights_path = "/home/tomasdu/repos/plastic_NNs/src/daCosta2024/RetinalSampling/CNN_model/model.h5"
        # weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260303_114100-f9orz4jp/files/model/best_model_full.pth"
        # weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260303_113834-arh3er1c/files/model/best_model_full.pth"
        # weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260303_124500-uuzfkh62/files/model/best_model_full.pth"
        # weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260303_130650-b66ud3uj/files/model/best_model_full.pth"
        # weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260303_130856-98ysm2u5/files/model/best_model_full.pth"
        # weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260303_134913-uv70xl1u/files/model/best_model_full.pth"
        # weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260304_104311-gfn3gqeq/files/model/best_model_full.pth"
        # weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260304_132159-qbmv1aue/files/model/best_model_full.pth"
        # lc_weights = "/home/tomasdu/repos/experiments/plastic_NNs/active/wang-ILSVRC_centered/WS-S/wandb/offline-run-20260215_150432-apv3qmwe/files/model/best_model_full.pth"
        # weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260305_171711-3c7h7pum/files/model/best_model_full.pth"
        # weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260305_173739-zo0akl5l/files/model/best_model_full.pth"
        # weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260305_175328-fvq0rekg/files/model/best_model_full.pth"
        # weights_path="/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260305_182415-os9zi3ym/files/model/best_model_full.pth"
        # below, conv on full ILSVRC
        # weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260308_135911-tmax5ra3/files/model/best_model_full.pth"
        # conv on 10 classes with aug
        # weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260313_132042-o2dd0zs7/files/model/best_model_full.pth"
        # conv on 10 classes with aug and norm after act
        # weights_path = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260313_140845-bbsedv10/files/model/best_model_full.pth"
        # lc_weights=  "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260313_202036-37nsu6q1/files/model/best_model_full.pth"
        # lc_weights = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260317_103554-9ybvy4nq/files/model/best_model_full.pth"
        # lc_weights = "/home/tomasdu/repos/experiments/plastic_NNs/active/fisheye_dws/WS-S/wandb/offline-run-20260313_190823-57ybbp12/files/model/best_model_full.pth"
