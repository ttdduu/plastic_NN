from src.utils.dataset_factory import DatasetFactory
from collections import Counter
import torch

class DataLoaderMixin:
    def _setup_data(self):
        """Setup data loaders with minimal overhead"""
        self.logger.log(f"Using {self.config.data.fraction * 100}% of the dataset")
        
        # Get dataset without immediate loading
        self.loaders, self.class_names, self.input_size = DatasetFactory.get_dataset(self.config)
        self.train_loader = self.loaders['train']
        self.val_loader = self.loaders['val']
        
        # Log basic info without computing distributions
        self._log_dataset_info()

        # Determine the (H, W) to size the model's layers. Under reconstruction
        # mode BOTH splits now yield the PRE-fisheye clean (val too, so
        # _recon_validate can reuse the training mask path), i.e. this sample is
        # 256², not the post-fisheye 156². That's fine: dws_mix re-applies the
        # fisheye to input_H, so input_H=256 → the post-fisheye 156² the LC needs
        # (same result as a 156² sample, since fe(256)=fe(156)=156). Non-recon
        # yields the warped 156² directly. Either way the built model is 156².
        try:
            sample_img, _ = self.val_loader.dataset[0]
            # Under recon the val sample is now PRE-fisheye (256²) so _recon_validate
            # can reuse the training mask path. But the model is sized DIRECTLY from
            # input_H/W (dws_mix does NOT re-apply the fisheye), so warp the sample to
            # the POST-fisheye size the model actually processes — else an HC model
            # builds its LC at 256² while every input is 156² (the 24336-vs-65536
            # crash). A conv control is size-agnostic and wouldn't notice.
            # fisheye_in_model: the MODEL warps, so input_H/W must stay the PRE-fisheye size the
            # model is actually fed. Warping the sample here would report the post-warp size, the
            # model would size its stem for a SECOND warp of that, and — because the runtime shape
            # guard only compares against that (wrong) size — it would warp twice without
            # complaining. So this branch is skipped entirely when the model owns the warp.
            if (not bool(getattr(self.config.model, 'fisheye_in_model', False))
                    and bool(getattr(self.config.training, 'recon_loss_enabled', False))):
                from src.training.reconstruction_loss import _recon_fisheye
                _fe = _recon_fisheye(self.config.data)
                if _fe is not None:
                    sample_img = _fe(sample_img.unsqueeze(0)).squeeze(0)
            h, w = sample_img.shape[-2], sample_img.shape[-1]
            self.config.model.input_H = int(h)
            self.config.model.input_W = int(w)
        except Exception:
            # If sampling fails, skip setting explicit input size
            pass
