from torchvision.transforms import v2


ToRGB = v2.Lambda(
    lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x
)
