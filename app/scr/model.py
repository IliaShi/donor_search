import torch
from torch import nn
import torchvision
from torchvision.models import resnet50
from torchvision import transforms
import PIL
import os

# Константы
CLASSES = {0:'0', 1:'090', 2:'180', 3:'270'}

# Определение модели
class VGGCLF(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.vgg11(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        layers = list(self.resnet.children())
        for layer in layers[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.fc = nn.Linear(1000, 4)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

def image_rotate(path: str, angle: int):
    ''' Функция поворачивает изображение в правильное положение с учетом ответа модели'''
    # Открываем файл
    try:
        image = PIL.Image.open(path)
    except PIL.UnidentifiedImageError:
        return 'Error loading image'

    # Поворачиваем изображение и сохраняем файл
    try:
        image_rot = image.rotate(angle=360-angle, expand=True)
        image_rot.save(path)
        image.close()
    except:
        return 'Error saving image'

    return 'OK'

def predict(path: str, inp_size: int, device_: str):
    ''' Функция загружает изображение и получает предсказание модели'''
    # Открываем файл
    try:
        image = PIL.Image.open(path).convert('RGB')
    except PIL.UnidentifiedImageError:
        return 'File type error'

    # Трансформации изображения перед загрузкой
    transform = transforms.Compose([
        transforms.Resize((inp_size, inp_size)),
        transforms.ToTensor()
        ])
    image = transform(image)

    # Загружаем модель
    model = VGGCLF()
    model.to(device_)
    model.load_state_dict(torch.load('src/models/model_rn.pth', map_location=torch.device(device_)))

    # Получаем прогноз модели
    with torch.no_grad():
        x = image.to(device_).unsqueeze(0)
        predictions = model.eval()(x)
    result = int(torch.argmax(predictions, 1).cpu())

    # Вызов функции для поворота изображения
    rot_result = image_rotate(path, int(CLASSES[result]))

    return rot_result, int(CLASSES[result])
