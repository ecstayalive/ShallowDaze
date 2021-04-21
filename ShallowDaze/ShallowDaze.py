import os
import subprocess
import sys
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from .creator.SirenCreator import SirenNet, SirenWrapper
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch_optimizer import DiffGrad, AdamP

from imageio import imread, mimsave
import torchvision.transforms as T

from tqdm import trange, tqdm  # 进度条模块

from .clip import load, tokenize


# Helpers

def exists(val):
    return val is not None  # val 不是None就返回


def default(val, d):
    return val if exists(val) else d  # 如果存在val就返回，否则返回d


def interpolate(image, size):
    """
    在模型接受的基础上提高图像分辨率
    """
    return F.interpolate(image, (size, size), mode='bilinear', align_corners=False)


def rand_cutout(image, size, center_bias=False, center_focus=2):
    width = image.shape[-1]
    min_offset = 0
    max_offset = width - size
    if center_bias:
        # 图像中心附近的样本
        center = max_offset / 2
        std = center / center_focus
        offset_x = int(random.gauss(mu=center, sigma=std))
        offset_y = int(random.gauss(mu=center, sigma=std))
        # 如果超出边界，则均匀地重新采样
        offset_x = random.randint(min_offset, max_offset) if (
                offset_x > max_offset or offset_x < min_offset) else offset_x
        offset_y = random.randint(min_offset, max_offset) if (
                offset_y > max_offset or offset_y < min_offset) else offset_y
    else:
        offset_x = random.randint(min_offset, max_offset)
        offset_y = random.randint(min_offset, max_offset)

    cutout = image[:, :, offset_x:offset_x + size, offset_y:offset_y + size]

    return cutout


def create_clip_img_transform(image_width):
    """
    对图像进行初始变换
    """
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    transform = T.Compose([
        T.Resize(image_width),
        T.CenterCrop((image_width, image_width)),
        T.ToTensor(),
        T.Normalize(mean=clip_mean, std=clip_std)
    ])
    return transform


def open_folder(path):
    # 获取当前文件路径
    if os.path.isfile(path):
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        return

    cmd_list = None
    if sys.platform == 'darwin':
        cmd_list = ['open', '--', path]
    elif sys.platform == 'linux2' or sys.platform == 'linux':
        cmd_list = ['xdg-open', path]
    elif sys.platform in ['win32', 'win64']:
        cmd_list = ['explorer', path.replace('/', '\\')]
    if cmd_list is None:
        return

    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass


def norm_siren_output(img):
    return ((img + 1) * 0.5).clamp(0.0, 1.0)


def create_text_path(context_length, text=None, img=None):
    if text is not None:
        input_name = "./pictures/" + text.replace(" ", "_")[:context_length]
    elif img is not None:
        if isinstance(img, str):
            input_name = "".join(img.replace(" ", "_").split(".")[:-1])
        else:
            input_name = "PIL_img"
    else:
        input_name = "your_encoding"
    return input_name


class ShallowDaze(nn.Module):
    def __init__(
            self,
            clip_perceptor,  # clip感知器
            clip_norm,  #
            input_res,  # 输入分辨率
            total_batches,  # 总共的批量
            batch_size,  # 批大小
            num_layers=8,
            image_width=400,
            loss_coef=100,
            theta_initial=None,
            theta_hidden=None,  # 色彩空间
            lower_bound_cutout=0.1,  # should be smaller than 0.8
            upper_bound_cutout=1.0,
            do_cutout=True,
            center_bias=False,
            center_focus=2,
            hidden_size=256,
            averaging_weight=0.3,  # 平均权重
    ):
        super().__init__()
        # 加载clip模型
        self.perceptor = clip_perceptor  # clip感受器
        self.input_resolution = input_res  # 输入分辨率
        self.normalize_image = clip_norm  # 归一化图像

        self.loss_coef = loss_coef
        self.image_width = image_width

        self.batch_size = batch_size
        self.total_batches = total_batches
        self.num_batches_processed = 0  # 批处理数量

        w0 = default(theta_hidden, 30.)
        w0_initial = default(theta_initial, 30.)

        # 具有周期激活函数的隐式神经表示，其效果要比RELU更好
        siren = SirenNet(
            dim_in=2,
            dim_hidden=hidden_size,  # 隐藏的维度
            num_layers=num_layers,  # 隐藏层数量
            dim_out=3,  # rgb
            use_bias=True,
            w0=w0,
            w0_initial=w0_initial
        )

        # 装饰器从给定的SirenNet中训练出特定高度和宽度的特定图像，然后再生成。
        # 调用装饰器后经过训练会生成图像
        self.model = SirenWrapper(
            siren,
            image_width=image_width,
            image_height=image_width
        )

        self.saturate_limit = 0.75  # 饱和极限，高于此值的切口会导致不稳定
        self.lower_bound_cutout = lower_bound_cutout
        self.upper_bound_cutout = upper_bound_cutout
        self.do_cutout = do_cutout
        self.center_bias = center_bias
        self.center_focus = center_focus
        self.averaging_weight = averaging_weight

    def sample_sizes(self, lower, upper, width):
        lower *= width
        upper *= width
        sizes = torch.randint(int(lower), int(upper), (self.batch_size,))
        return sizes

    def forward(self, text_embed, return_loss=True, dry_run=False):
        out = self.model()  # 通过sirennet训练出特定高度的图像

        out = norm_siren_output(out)  # 将输入 压缩值0到1之间
        # 如果不要求训练，则只是单纯的返回图像
        if not return_loss:
            return out
        # 确定上限和下限采样范围
        width = out.shape[-1]
        lower_bound = self.lower_bound_cutout  # 下限切口
        # 下限和上限之间的样本切口大小
        sizes = self.sample_sizes(lower_bound, self.upper_bound_cutout, width)
        # 创建标准化的随机切口
        if self.do_cutout:
            image_pieces = [rand_cutout(out, size, center_bias=self.center_bias, center_focus=self.center_focus) for
                            size in sizes]
            image_pieces = [interpolate(piece, self.input_resolution) for piece in image_pieces]
        else:
            image_pieces = [interpolate(out.clone(), self.input_resolution) for _ in sizes]

        # 归一化
        # torch.cat矩阵拼接，默认为：0：竖着凭借，1：横着拼接
        image_pieces = torch.cat([self.normalize_image(piece) for piece in image_pieces])

        # 计算图像嵌入
        # 充当上下文管理器或修饰器，使您的脚本区域可以混合精度运行
        with autocast(enabled=False):
            # 图像嵌入，使用clip模型为图像编码
            image_embed = self.perceptor.encode_image(image_pieces)
        # 计算损失
        # loss over averaged features of cutouts
        # 平均特征丢失 torch.unsqueeze:返回一个新的张量，对输入的既定位置插入维度 1
        avg_image_embed = image_embed.mean(dim=0).unsqueeze(0)

        # 平均损失
        # cosine_similarity函数对两个向量或者张量计算余弦相似度
        # 余弦相似度用向量空间中两个向量夹角的余弦值作为衡量两个个体间差异的大小。
        # 相比距离度量，余弦相似度更加注重两个向量在方向上的差异，而非距离或长度上。
        averaged_loss = -self.loss_coef * torch.cosine_similarity(text_embed, avg_image_embed, dim=-1).mean()
        # loss over all cutouts
        # 所有切口的损失
        general_loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean()
        # merge losses
        # 合并亏损
        loss = averaged_loss * (self.averaging_weight) + general_loss * (1 - self.averaging_weight)

        # 计算批处理数量
        if not dry_run:
            self.num_batches_processed += self.batch_size
        # 返回模型和loss
        return out, loss


class Imagine(nn.Module):
    def __init__(
            self,
            *,
            text=None,  # 文本
            img=None,  # 想象的艺术图片
            lr=1e-5,  # 学习率
            batch_size=4,  #
            gradient_accumulate_every=4,  # 梯度累积，增大可以在比较小的epoch上快速降低loss
            save_every=100,  # 每迭代100次就保存一次
            image_width=200,  # 最大400，相应的layer最大14
            num_layers=8,
            epochs=3,
            iterations=1050,
            save_progress=True,
            open_folder=True,
            theta_initial=None,  # 描述siren初始层的色彩空间
            theta_hidden=None,  # 描述siren隐藏层的色彩空间
            model_name="ViT-B/32",  # 模型名称 VIT-B 小模型
            lower_bound_cutout=0.1,  # should be smaller than 0.8
            upper_bound_cutout=1.0,
            averaging_weight=0.3,
            do_cutout=True,
            center_bias=False,
            center_focus=2,
            optimizer="AdamP",
            jit=True,
            hidden_size=256,
            save_gif=True,
            save_video=True,
    ):

        super().__init__()

        self.epochs = epochs

        # jit models only compatible with version 1.7.1
        if "1.7.1" not in torch.__version__:
            if jit:
                print("Setting jit to False because torch version is not 1.7.1.")
            jit = False

        # 加载CLIP模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 如果gpu可用则选择，否则cpu
        # 在线下载神经网络ViT-B/32模型，供clip使用
        # 返回clip模型，nn.Module
        # Torchvision转换，将PIL图像转换为张量，返回的模型可以将其用作输入
        clip_perceptor, norm = load(model_name, jit=jit, device=self.device)
        # 不启用 Batch Normalization 和 Dropout。
        # 生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。
        self.perceptor = clip_perceptor.eval()
        for param in self.perceptor.parameters():
            param.requires_grad = False

        if not jit:
            input_res = clip_perceptor.visual.input_resolution  # 输入分辨率
        else:
            input_res = clip_perceptor.input_resolution.item()
        # 创建clip图像transform模型
        self.clip_transform = create_clip_img_transform(input_res)
        # 迭代次数
        self.iterations = iterations
        # 生成图像的宽度
        self.image_width = image_width
        # 总共的批大小 = imagine输入的epochs（20） * 迭代次数 * batch的大小 * 梯度累积
        total_batches = self.epochs * self.iterations * batch_size * gradient_accumulate_every
        # 加载DeepDaze模型，并将该模型部署到gpu或cpu上
        model = ShallowDaze(
            self.perceptor,  # clip模型
            norm,  # clip模型范数
            input_res,  # 输入分辨率
            total_batches,
            batch_size=batch_size,  # batch_size=4
            image_width=image_width,
            num_layers=num_layers,
            theta_initial=theta_initial,  # None
            theta_hidden=theta_hidden,  # None
            lower_bound_cutout=lower_bound_cutout,  # 0.1
            upper_bound_cutout=upper_bound_cutout,  # 1.0
            do_cutout=do_cutout,
            center_bias=center_bias,
            center_focus=center_focus,
            hidden_size=hidden_size,
            averaging_weight=averaging_weight,
        ).to(self.device)
        self.model = model  # deep-daze模型
        self.scaler = GradScaler()  # 通过放大loss的值来防止梯度的下溢
        siren_params = model.model.parameters()
        # 三种梯度下降的方法，默认AdamP
        if optimizer == "AdamP":
            self.optimizer = AdamP(siren_params, lr)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(siren_params, lr)
        elif optimizer == "DiffGrad":
            self.optimizer = DiffGrad(siren_params, lr)

        # 梯度累积
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every
        self.open_folder = open_folder
        self.save_progress = save_progress
        self.text = text
        self.image = img
        # 默认clip_encoding=None
        self.textpath = create_text_path(self.perceptor.context_length, text=text, img=img)
        self.filename = self.image_output_path()

        # 创建代码以进行优化
        self.clip_encoding = self.create_clip_encoding(text=text, img=img)  # 默认clip_encoding=None

        self.save_gif = save_gif
        self.save_video = save_video

    def create_clip_encoding(self, text=None, img=None):
        self.text = text
        self.img = img
        if text is not None and img is not None:
            encoding = (self.create_text_encoding(text) + self.create_img_encoding(img)) / 2
        elif text is not None:
            encoding = self.create_text_encoding(text)
        elif img is not None:
            encoding = self.create_img_encoding(img)
        return encoding

    def create_text_encoding(self, text):
        """
        利用clip模型创建text的token
        """
        tokenized_text = tokenize(text).to(self.device)
        with torch.no_grad():
            text_encoding = self.perceptor.encode_text(tokenized_text).detach()
        return text_encoding

    def create_img_encoding(self, img):
        """
        通过clip模型编码图像
        """
        normed_img = self.clip_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_encoding = self.perceptor.encode_image(normed_img).detach()
        return img_encoding

    def set_clip_encoding(self, text=None, img=None, encoding=None):
        """
        通过clip模型将编码好的图像和文本组合起来
        """
        encoding = self.create_clip_encoding(text=text, img=img, encoding=encoding)
        self.clip_encoding = encoding.to(self.device)

    def image_output_path(self, sequence_number=None):
        """
        返回下划线分隔的Path
          如果设置了“ self.save_date_time”，则以当前时间戳为准
          如果设置了“ save_every”，则在序列号的左边填充6个零
        :rtype: Path
        """
        output_path = self.textpath
        if sequence_number:
            sequence_number_left_padded = str(sequence_number).zfill(6)
            output_path = f"{output_path}.{sequence_number_left_padded}"
        return Path(f"{output_path}.jpg")

    def train_step(self, epoch, iteration):
        """
        epoch = 3, iteration = 1050
        @return: 权值和loss
        """
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):  # gradient_accumulate_every=4
            # 充当上下文管理器或修饰器，使您的脚本区域可以混合精度运行
            with autocast(enabled=True):
                out, loss = self.model(self.clip_encoding)  # 通过deep-daze模型训练图像与文本的联合编码
            # 计算损失
            loss = loss / self.gradient_accumulate_every
            total_loss += loss

            self.scaler.scale(loss).backward()  # 反向梯度

        out = out.cuda().float().clamp(0., 1.)
        self.scaler.step(self.optimizer)
        self.scaler.update()  # 按照优化器更新权值
        self.optimizer.zero_grad()  # 每次训练将梯度累积，否则会影响下一次梯度计算

        if (iteration % self.save_every == 0) and self.save_progress:
            self.save_image(epoch, iteration, img=out)

        return out, total_loss

    def get_img_sequence_number(self, epoch, iteration):
        current_total_iterations = epoch * self.iterations + iteration
        sequence_number = current_total_iterations // self.save_every
        return sequence_number

    @torch.no_grad()
    def save_image(self, epoch, iteration, img=None):
        sequence_number = self.get_img_sequence_number(epoch, iteration)

        if img is None:
            img = self.model(self.clip_encoding, return_loss=False).cpu().float().clamp(0., 1.)
        self.filename = self.image_output_path(sequence_number=sequence_number)

        pil_img = T.ToPILImage()(img.squeeze())
        pil_img.save(self.filename, quality=95, subsampling=0)
        pil_img.save(f"{self.textpath}.jpg", quality=95, subsampling=0)

        tqdm.write(f'image updated at "./{str(self.filename)}"')

    def generate_gif(self):
        images = []
        for file_name in sorted(os.listdir('./')):
            if file_name.startswith(self.textpath) and file_name != f'{self.textpath}.jpg':
                images.append(imread(os.path.join('./', file_name)))

        if self.save_video:
            mimsave(f'{self.textpath}.mp4', images)
            print(f'Generated image generation animation at ./{self.textpath}.mp4')
        if self.save_gif:
            mimsave(f'{self.textpath}.gif', images)
            print(f'Generated image generation animation at ./{self.textpath}.gif')

    def forward(self):

        tqdm.write(f'Imagining "{self.textpath}" from the depths of my weights...')

        with torch.no_grad():
            self.model(self.clip_encoding, dry_run=True)  # do one warmup step due to potential issue with CLIP and CUDA
        # 打开文件夹
        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        try:
            for epoch in trange(self.epochs, desc='epochs'):  # self.epochs = 3
                pbar = trange(self.iterations, desc='iteration')  # self.iterations = 1050
                for i in pbar:
                    _, loss = self.train_step(epoch, i)  # 训练
                    pbar.set_description(f'loss: {loss.item():.2f}')  # 进度条设置

        except KeyboardInterrupt:
            print('interrupted by keyboard, gracefully exiting')
            return

        self.save_image(epoch, i)  # one final save at end

        if (self.save_gif or self.save_video) and self.save_progress:
            self.generate_gif()
