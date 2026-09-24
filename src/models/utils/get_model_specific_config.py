def get_model_specific_config(self):
    return {
        "conv_weights": self.conv_weights,
        "lc_weights": self.lc_weights,
        "num_classes": int(getattr(self.config, "num_classes", 15)),
        "no_stem": self.no_stem,
        "recurrent_timesteps": self.T,
        "lateral_kernel_size": self.lateral_kernel_size,
        "lateral_init_mode": self.lateral_init_mode,
        "lateral_init_scale": self.lateral_init_scale,
        "recurrent_norm_mode": self.recurrent_norm_mode,
        "lateral_target": self.lateral_target,
    }



