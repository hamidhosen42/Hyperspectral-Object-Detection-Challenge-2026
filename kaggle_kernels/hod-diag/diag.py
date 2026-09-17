import os, subprocess
print(subprocess.run('ls -la /kaggle/input; find /kaggle/input -maxdepth 4 | head -50; df -h /kaggle/working /kaggle/tmp 2>/dev/null; nvidia-smi -L; nproc; free -g', shell=True, capture_output=True, text=True).stdout)
