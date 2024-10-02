# Скрипт содержит функции для обучения моделей и оценки метрик

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

def fit_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """Функция выполняет итерацию обучения в рамках одной эпохи, вычисляет функцию потерь,
        обновляет параметры модели, отслеживает метрику
    Args:
        model - модель
        train_loader - загрузчик для трейна
        criterion - функция потерь
        optimizer - оптимайзер
        scheduler - шедулер
        device - девайс
    Returns:
        train_loss - потери
        train_acc - accuracy
    """
    model.train()   # Перевод модели в режим обучения

    # Переменные для отслеживания ошибочных и правильных предсказаний, количества обраобтанных данных
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for inputs, labels in train_loader:
        # Перенос данных на устройство
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()               # Сброс градиентов параметров модели перед новым шагом обр. распростр.ошибки
        outputs = model(inputs)             # Расчет выводов модели
        loss = criterion(outputs, labels)   # Вычисление ошибки
        loss.backward()                     # Расчет градиентов по параметрам модели
        optimizer.step()                    # Обновление параметров модели
        preds = torch.argmax(outputs, 1)    # Выбор индекса с макс.значением, т.е. определение ответ модели
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)

    scheduler.step()                        # Обновление шедулера после окончания эпохи
    train_loss = running_loss / processed_data  # Расчет средней потери по эпохе
    train_acc = running_corrects.cpu().numpy() / processed_data # Расчет accuracy эпохи

    return train_loss, train_acc

def eval_epoch(model, val_loader, criterion, device):
    """ Функция оценивает точность модели после очередной эпохи на валидационном сете
    Args:
        model - модель
        val_loader - загрузчик для валидации
        criterion - ф-я потерь
    Returns:
        train_loss - потери
        train_acc - accuracy
        device - девайс
    """
    model.eval()    # Перевод модели в режим вывода

    # Переменные для отслеживания ошибочных и правильных предсказаний, количества обраобтанных данных
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for inputs, labels in val_loader:
        # Перенос данных на устройство
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):         # Отключение вычисления градиентов
            outputs = model(inputs)                 # Расчет выводов модели
            loss = criterion(outputs, labels)       # Вычисление ошибки
            preds = torch.argmax(outputs, 1)        # Выбор индекса с макс.значением, т.е. определение ответа модели

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)

    val_loss = running_loss / processed_data        # Расчет средней потери по эпохе
    val_acc = running_corrects.cpu().numpy() / processed_data # Расчет accuracy эпохи

    return val_loss, val_acc

def train_model(train, val, model, epochs, rate, criterion, device):
    """Функция осуществляет обучение модели
    Args:
        train - загрузчик для тренировочного сета
        val - загрузчик для валидационного сета
        model - экземпляр модели
        epochs - кол-во эпох для обучения
        rate - learning rate оптимизатора
        criterion - ф-я потерь
        device - девайс
    Returns:
        history - история обучения: потери и accuracy на каждой эпохе
    """
    history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f}  \
                val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        opt = torch.optim.Adam(model.parameters(), lr=rate)
        sch = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
        for epoch in range(epochs):
            train_loss, train_acc = fit_epoch(model, train, criterion, opt, sch, device=device)

            val_loss, val_acc = eval_epoch(model, val, criterion, device=device)
            history.append((train_loss, train_acc, val_loss, val_acc))

            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss,\
                                           v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))
    return history

def training_visualisation(history, model_name=''):
    """ Функция визуализирует функцию потерь и accuracy на каждой эпохе
    Args:
        history - история обучения: потери и accuracy на каждой эпохе
        model_name (str) - название модели
    Returns: нет
    """
    train_loss, train_acc, val_loss, val_acc = zip(*history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # отрисовка потерь
    axes[0].plot(train_loss, label='train_loss')
    axes[0].plot(val_loss, label='val_los')
    axes[0].set_xlabel('Эпохи')
    axes[0].set_ylabel('Потери')
    axes[0].set_title(model_name + ': потери')
    axes[0].legend()

    # отрисовка accuracy
    axes[1].plot(train_acc, label='train_acc')
    axes[1].plot(val_acc, label='val_acc')
    axes[1].set_xlabel('Эпохи')
    axes[1].set_ylabel('accuracy')
    axes[1].set_title(model_name + ': accuracy')
    axes[1].legend(loc='best')

def metrics_evaluation(model, test_loader, device, draw_graph=False):
    """Функция расчитывает метрики accuracy, precision, recall, f1, auc;
        Визуализирует матрицу ошибок и ROC-кривые
    Args:
        model - модель
        test_loader - загрузчик тестовых данных
        device - девайс
        draw_graph - если True - отрисовывает ROC-кривые и матрицу ошибок
    Returns:
        metrics = {'accuracy', 'precision', 'recall', f1, auc}
    """
    model.eval() # переводим модель в режим вывода

    y_true = []
    y_pred = []
    probabilities_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Перенос данных на устройство
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)                     # Расчет выводов модели
            _, predicted = torch.max(outputs.data, 1)   # Опрделение класса с наибольшей вероятностью
            y_true.extend(labels.cpu().numpy())         # Преобразование меток теста в список y_true
            y_pred.extend(predicted.cpu().numpy())      # Преобразование предсказаний модели в список y_pred
            probabilities_list.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()) # преобразование в список вероятностей

            # Расчет базовых метрик (используется микроусреднение)
            acc = float(round(accuracy_score(y_true, y_pred), 3))
            prec = float(round(precision_score(y_true, y_pred, average='micro'), 3))
            recall = float(round(recall_score(y_true, y_pred, average='micro'), 3))
            f1 = float(round(f1_score(y_true, y_pred, average='micro'), 3))

    # Визуализация матрицы ошибок
    if draw_graph:
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.title('Матрица ошибок')
        plt.show()

    # Расчет и визуализация ROC-AUC
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    probabilities_list = np.array(probabilities_list)
    if draw_graph:
        plt.figure(figsize=(8, 6))
    colors = ['darkorange', 'darkgreen', 'darkblue', 'darkred']
    for i in range(4):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probabilities_list[:, i])
        roc_auc = float(round(auc(fpr, tpr), 3))
        if draw_graph:
            plt.plot(fpr, tpr, color=colors[i], label=f'Класс {i} AUC: {roc_auc}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC-кривые для классов')
            plt.legend(loc="lower right")

    metrics = {'accuracy': acc,
               'precision': prec,
               'recall': recall,
               'f1': f1,
               'AUC': roc_auc
                }
    return metrics

def model_inference_eval(model, image, device, num_iter = 100):
    """ Функция замеряет скорость обработки изображения моделью
    Args:
        model - обученная модель
        image - изображение
        device - девайс
        num_iter - количество итераций
    Returns:
        fpc - время обработки одного изображения
    """
    model.eval()
    model = model.to(device)
    image = image.to(device)

    start_time = time.time()
    for _ in range(num_iter):
        with torch.no_grad():
            _ = model(image)
    end_time = time.time()
    total_time = end_time - start_time
    fps = round(total_time / num_iter, 4)

    return fps
