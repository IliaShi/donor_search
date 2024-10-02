from fastapi import FastAPI, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import argparse
import uvicorn
from model import predict
import os

# Константы
DEVICE = "cpu"
INP_SIZE = 128

# Веб-интерфейс
app = FastAPI()

# Создаем папку tmp, если она не существует
if not os.path.exists('tmp'):
    os.makedirs('tmp')

# Определяем папку для статических файлов
app.mount("/tmp", StaticFiles(directory= "tmp"), name='images')

# Определяем шаблонизатор и назначаем папку с html-шаблонами
templates = Jinja2Templates(directory='src/templates')

@app.get('/health')
def health():
    return {'status':'OK'}

@app.get('/')
def main(request:Request):
    """Возвращает start_form"""
    return templates.TemplateResponse('start_form.html', {'request':request})

@app.post('/predict')
def temp(file: UploadFile, request: Request):
    """ Обрабатывает нажатие кнопки "Загрузить" на Start_form"""
    # Сохранение загружаемого файла в tmp
    save_path = 'tmp/' + file.filename
    with open(save_path, 'wb') as fid:
        fid.write(file.file.read())
    # Вызов функции с моделью
    result = predict(save_path, inp_size=INP_SIZE, device_=DEVICE)
    message = result[0]

    if result[0] == 'OK':
        return templates.TemplateResponse('detect_form.html',
                                        {'request': request,
                                         'res': result[1],
                                         'path':save_path,
                                         'message': message
                                         })
    else:
        return templates.TemplateResponse('error_form.html',
                                        {'request': request,
                                         'message': result,
                                         'path':save_path,
                                         })

# Запуск приложения локально
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8000, type=int, dest='port')
    parser.add_argument('--host', default='0.0.0.0', type=str, dest='host')
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
