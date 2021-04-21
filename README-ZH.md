# ShallowDaze

## 介绍
ShallowDaze是可以从文本生成图像的脚本。建立在deep-daze的基础上,更改了其网络参数，目前尝试添加一些用于生成图片的网络模型，并删除了一些内容。 安装环境后，默认情况下它可以在视频内存少于4GB的图形卡上运行。 如果您想了解有关深度发呆的更多详细信息，
 请到原项目[deep-daze](https://github.com/lucidrains/deep-daze) 中查看。 这个作者简直了！！

## 例子

*五彩斑斓的黑*

![Colorful black](./samples/Colorful_black.gif)

*梵高的原作星空*

![Van Gogh's original The Starry Night](./samples/Van_Gogh's_original_The_Starry_Night.gif)

*浩瀚的海洋，如云的山脉，夕阳染红了天空*

![The vast ocean, the mountains into the clouds, the sunset stained the sky](./samples/The_vast_ocean_the_mountains_into_the_clouds_the_sunset_stained_the_sky.gif)

*一颗用脚跑步的树*

![A running tree with its feet](./samples/A_running_tree_with_its_feet.gif)

*有翅膀的老虎飞翔在天空上*

![A tiger with wings is flying on the blue sky](./samples/A_tiger_with_wings_is_flying_on_the_blue_sky.gif)

*海洋*

![ocean](./samples/ocean.gif)

*森林*

![forest](./samples/forest.gif)

*夜空布满了星星，月亮挂在上面*

![The night sky is full of stars, and the moon is hang on there](./samples/The_night_sky_is_full_of_stars_and_the_moon_is_hang_on_there.gif)

## 如何使用?

在运行脚本前，先配置好环境
```
git clone https://github.com/ecstayalive/ShallowDaze
cd ShallowDaze
python main.py
```

## 引文

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```

```bibtex
@misc{sitzmann2020implicit,
    title   = {Implicit Neural Representations with Periodic Activation Functions},
    author  = {Vincent Sitzmann and Julien N. P. Martel and Alexander W. Bergman and David B. Lindell and Gordon Wetzstein},
    year    = {2020},
    eprint  = {2006.09661},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```