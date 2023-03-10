import torchvision.models as models

# import torchvision.models.quantization as models
#
# model = models.googlenet(pretrained=True, progress=True, quantize=True)

model = models.alexnet(pretrained=True, progress=True)
# print(model.pretrained)

print(model)

# for param in model.parameters():
#     param.requires_grad = False

# model.classifier[6].requires_grad_(False)
for param in model.classifier[6].parameters():
    param.requires_grad = False
count = 0
# without any trainable parameters . what's the fuck
for param in model.parameters():
    count += 1
    print(param.requires_grad)

print(count)
