import json
import numpy as np
import matplotlib.pyplot as plt
import os

json_file="scaling_results.json"

if not os.path.exists(json_file):
    raise FileNotFoundError(f"please run train_scaling,py")

with open(json_file,"r") as f:
    result=json.load(f)
    
N_values=np.array([metrics['N'] for metrics in result['parameter_scaling'].values()])
L_N_values = np.array([metrics["L"] for metrics in result["parameter_scaling"].values()])

D_values = np.array([metrics["D"] for metrics in result["data_scaling"].values()])
L_D_values = np.array([metrics["L"] for metrics in result["data_scaling"].values()])

log_N = np.log10(N_values)
log_L_N = np.log10(L_N_values)

log_D = np.log10(D_values)
log_L_D = np.log10(L_D_values)

slope_N, intercept_N = np.polyfit(log_N, log_L_N, 1)
alpha_N = -slope_N

slope_D, intercept_D = np.polyfit(log_D, log_L_D, 1)
alpha_D = -slope_D

gamma = alpha_N / alpha_D

print(f"Alpha_N (Parameter Exponent): {alpha_N}")
print(f"Alpha_D (Data Exponent)     : {alpha_D}")
print(f"Gamma (Scaling Ratio)       : {gamma}")


fig , (ax1,ax2)=plt.subplot(1,2,figsize=(14,6))
ax1.scatter(N_values,L_N_values,color='blue',s=100,zorder=5)
fit_L_N=10**(slope_N*log_N+intercept_N)
ax1.plot(N_values,fit_L_N,color='red',linestyle='-')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('Parameter Scaling Law', fontsize=14, fontweight='bold')
ax1.set_xlabel('Non-Embedding Parameters', fontsize=12)
ax1.set_ylabel('Validation Loss', fontsize=12)
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend(fontsize=12)


ax2.scatter(D_values, L_D_values, color='green', s=100, zorder=5)
fit_L_D = 10 ** (slope_D * log_D + intercept_D)
ax2.plot(D_values, fit_L_D, color='orange', linestyle='--')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('Data Scaling Law', fontsize=14, fontweight='bold')
ax2.set_xlabel('Dataset Size in Tokens', fontsize=12)
ax2.set_ylabel('Validation Loss', fontsize=12)
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig('scaling_laws.png', dpi=300, bbox_inches='tight')

