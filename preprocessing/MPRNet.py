import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import shutil  
from runpy import run_path
from skimage import img_as_ubyte
from skimage.restoration import estimate_sigma  # 用于估计高斯噪声
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2

# 指定路径
input_dir = r'input_path'
result_dir = r'output_path'
task = 'Denoising'  # 'Deblurring', 'Denoising', 'Deraining'

#  保存图像并自动创建文件夹路径
def save_img(filepath, img):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

#  加载模型权重
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

#  估计图像的高斯噪声标准差
def get_estimated_sigma(image):
    return estimate_sigma(image, channel_axis=None) * 255

#  设置输入输出路径
inp_dir = input_dir
out_dir = result_dir
os.makedirs(out_dir, exist_ok=True)

#  读取所有图片，包括子文件夹
files = natsorted(glob(os.path.join(inp_dir, '**', '*.jpg'), recursive=True) +
                  glob(os.path.join(inp_dir, '**', '*.png'), recursive=True))
if len(files) == 0:
    raise Exception(f"No image files found at {inp_dir}")

#  设置设备（GPU 优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  加载模型及预训练权重
load_file = run_path(os.path.join(task, "MPRNet.py"))
model = load_file['MPRNet']().to(device).half()
weights = os.path.join(task, "pretrained_models", "model_" + task.lower() + ".pth")
load_checkpoint(model, weights)
model.eval()

#  定义输入图像的预处理：统一尺寸并转为张量
target_size = (512, 512)
transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor()
])

batch_size = 1
batched_inputs = []
file_names = []
variance_log_dict = {}

for file_ in files:
    img = Image.open(file_).convert('RGB')
    img_cv2 = cv2.imread(file_)
    img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    #  估计高斯噪声标准差
    estimated_sigma = get_estimated_sigma(img_gray)

    #  计算相对路径，保持输出目录结构
    relative_path = os.path.relpath(file_, inp_dir)
    output_img_path = os.path.join(out_dir, os.path.splitext(relative_path)[0] + '.png')
    output_log_path = os.path.join(out_dir, os.path.dirname(relative_path), "noise_sigma_log.txt")

    if output_log_path not in variance_log_dict:
        variance_log_dict[output_log_path] = []


    #  判断是否需要去噪：sigma > 120
    # if estimated_sigma <= 120:
        # os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        # shutil.copy(file_, os.path.join(out_dir, relative_path))
        # variance_log_dict[output_log_path].append(
            # f"{relative_path}: Sigma={estimated_sigma:.2f}, Denoised=False\n")
        # continue

    #  添加当前图像到 batch
    input_ = transform(img).unsqueeze(0).to(device).half()
    batched_inputs.append(input_)
    file_names.append((file_, relative_path, output_img_path, output_log_path, estimated_sigma))

    #  如果达到 batch size 或最后一张图像，执行模型推理
    if len(batched_inputs) == batch_size or file_ == files[-1]:
        if batched_inputs:
            batched_inputs = torch.cat(batched_inputs, dim=0)

            with torch.no_grad():
                restored = model(batched_inputs)

            if isinstance(restored, list):
                restored = restored[0]

            restored = torch.clamp(restored, 0, 1)

            for idx, (file_, relative_path, output_img_path, output_log_path, estimated_sigma) in enumerate(file_names):
                restored_img = restored[idx].permute(1, 2, 0).cpu().numpy()
                restored_img = img_as_ubyte(restored_img)

                save_img(output_img_path, restored_img)
                variance_log_dict[output_log_path].append(
                    f"{relative_path}: Sigma={estimated_sigma:.2f}, Denoised=True\n")

            #  清空 batch
            batched_inputs = []
            file_names = []
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

#  将每个子目录中的日志写入对应 log 文件
for log_path, lines in variance_log_dict.items():
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.writelines(lines)

print(f" 高斯噪声标准差估计完成，结果已保存到 {out_dir}")
print(f" 对所有图像进行了 MPRNet 去噪")
print(f" 去噪后的文件已保存到 {out_dir}")
