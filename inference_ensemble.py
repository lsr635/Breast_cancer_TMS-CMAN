import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict
import json

class EnsembleInference:
    """模型集成推理"""
    def __init__(self, model_configs: List[Dict], device='cuda'):
        self.device = device
        self.models = self._load_models(model_configs)
        
    def _load_models(self, configs):
        models = []
        for config in configs:
            model = TMS_CMAN(**config['model_params'])
            checkpoint = torch.load(config['checkpoint_path'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            models.append(model)
        return models
    
    def predict(self, frames, subtraction=None, use_tta=True):
        """集成预测"""
        all_predictions = []
        
        for model in self.models:
            if use_tta:
                tta_preds = []
                for aug_fn in [self._identity, self._hflip, self._vflip]:
                    aug_frames = aug_fn(frames)
                    with torch.no_grad():
                        outputs = model(aug_frames, subtraction)
                        pred = torch.sigmoid(outputs['logits'])
                    tta_preds.append(pred)
                
                model_pred = torch.stack(tta_preds).mean(dim=0)
            else:
                with torch.no_grad():
                    outputs = model(frames, subtraction)
                    model_pred = torch.sigmoid(outputs['logits']) 
            all_predictions.append(model_pred)

        ensemble_pred = torch.stack(all_predictions).log().mean(dim=0).exp()
        return ensemble_pred
    
    def _identity(self, x):
        return x
    
    def _hflip(self, x):
        return torch.flip(x, dims=[-1])
    
    def _vflip(self, x):
        return torch.flip(x, dims=[-2])

def calibrate_predictions(logits, temperature=1.5):
    """温度缩放校准"""
    return logits / temperature

if __name__ == '__main__':
    model_configs = [
        {
            'model_params': {
                'num_classes': 2,
                'num_frames': 9,
                'cnn_backbone': 'convnext_tiny',
                'transformer_backbone': 'swin_tiny_patch4_window7_224',
                'use_subtraction': True
            },
            'checkpoint_path': 'checkpoints/model1_best.pth'
        },
        {
            'model_params': {
                'num_classes': 2,
                'num_frames': 9,
                'cnn_backbone': 'convnext_small',
                'transformer_backbone': 'swin_small_patch4_window7_224',
                'use_subtraction': False
            },
            'checkpoint_path': 'checkpoints/model2_best.pth'
        }
    ]
    
    # 创建集成推理器
    ensemble = EnsembleInference(model_configs)
    
    # 推理示例
    # frames = load_test_data()
    # predictions = ensemble.predict(frames, use_tta=True)