import time
import torch

print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('cuda version', torch.version.cuda)
print('device count', torch.cuda.device_count())
if torch.cuda.is_available():
    try:
        print('device name', torch.cuda.get_device_name(0))
    except Exception as e:
        print('failed to get device name:', e)

    # quick GPU op
    x = torch.randn(2048, 2048, device='cuda')
    t0 = time.time()
    y = x * 2
    torch.cuda.synchronize()
    t1 = time.time()
    print(f'GPU op time: {t1 - t0:.6f} s')
else:
    print('Running on CPU; performing small CPU op')
    x = torch.randn(2048, 2048)
    t0 = time.time()
    y = x * 2
    t1 = time.time()
    print(f'CPU op time: {t1 - t0:.6f} s')
