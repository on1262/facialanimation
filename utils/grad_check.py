import torch
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import figure



class GradCheck():
    def __init__(self, model, model_name, plot, plot_folder=None):
        self.plot = plot
        self.eps = 1e-10
        if plot:
            self.plot_path = os.path.join(plot_folder, model_name + '.png')
        self.model = model
        self.init_avg = {}
        for n, p in self.model.named_parameters(recurse=True):
            if p.requires_grad and ("bias" not in n):
                self.init_avg[n] = torch.log10(p.abs().mean() + self.eps).item()
    
    def check_grad(self, disp):
        avg_grad = {}
        avg_p = {}
        for n, p in self.model.named_parameters(recurse=True):
            if p.requires_grad and ("bias" not in n) and p.grad is not None:
                avg_grad[n] = p.grad.abs().mean().item()
                avg_p[n] = p.abs().mean().item()
        layers = avg_grad.keys()
        if disp:
            torch.set_printoptions(sci_mode=True)
            for n in layers:
                print(n, ' p=%.4e' % avg_p[n], ' grad=%.4e' % avg_grad[n])
        if self.plot:
            fig, _ = plt.subplots(figsize=(15, 20), dpi=160)
            fig.subplots_adjust(bottom=0.3)
            
            avg_grad = [math.log10(avg_grad[n]+self.eps) for n in layers]
            avg_p = [math.log10(avg_p[n]+self.eps) for n in layers]
            init_p = [self.init_avg[n] for n in layers]

            plt.bar(np.arange(len(avg_grad)), avg_grad, alpha=0.4, lw=1, color="c")
            plt.bar(np.arange(len(avg_p)), avg_p, alpha=0.4, lw=1, color="b")
            plt.bar(np.arange(len(init_p)), init_p, alpha=0.4, lw=1, color="g")
            plt.hlines(0, 0, len(avg_grad)+1, lw=2, color="k" )
            plt.xticks(range(0,len(avg_grad), 1), layers, rotation="vertical",fontsize='xx-small')
            plt.xlim(left=0, right=len(avg_grad))
            plt.ylim(bottom = -10, top=1) # zoom in on the lower gradient regions
            plt.xlabel("Layers", labelpad=10)
            plt.ylabel("gradient")
            plt.title("Gradient flow")
            plt.grid(True)
            
            plt.legend([Line2D([0], [0], color="c", lw=4),
                        Line2D([0], [0], color="b", lw=4),
                        Line2D([0], [0], color="g", lw=4),
                        ], ['avg-gradient', 'avg-params', 'init-params'])
            
            plt.savefig(self.plot_path)
            plt.close()
        
