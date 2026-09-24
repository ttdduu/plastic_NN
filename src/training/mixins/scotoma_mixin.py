from src.scotoma import ScotomaApplier

class ScotomaMixin:
    def _setup_scotoma(self):
        self.scotoma_applier = ScotomaApplier(self.config)  # Just pass the config
        
        if self.config.data.scotoma_apply:
            self._visualize_scotoma_examples()
    
    def _apply_scotoma(self, inputs):
        scotomized = self.scotoma_applier.apply_scotoma(
            inputs,
            method=self.config.data.scotoma_method,
            strength=self.config.data.scotoma_strength,
            radius=self.config.data.scotoma_radius,
            sharpness=self.config.data.scotoma_sharpness
        )
        self._debug_scotoma_application(scotomized, "Scotoma Application")
        return scotomized

    def _debug_scotoma_application(self, images, phase="unknown"):
        """Debug method to print image statistics"""
        pass