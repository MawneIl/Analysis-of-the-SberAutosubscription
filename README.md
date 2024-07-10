## Описание проекта
<em>
Обучение модели, предсказывающей целевое действие, на основе предоставленных данных о визитах на сайт сервиса "СберАвтоподписка".<br>
Для получения предсказаний, модель обернута в localhost web app, с помощью фреймворка FastApi.<br>
Реализована валидация входных/выходных данных, посредством библиотеки pydantic.<br>
</em>

## Использование
1. Откройте терминал Bash.
2. Клонируйте репозиторий проекта.<blockquote>git clone https://github.com/MawneIl/Analysis-of-the-SberAutosubscription.git</blockquote>
3. Перейдите в корневой каталог проекта.<blockquote>cd Analysis-of-the-SberAutosubscription</blockquote>
4. Установите необходимые библиотеки, выполнив команду: <blockquote>pip install requirements.txt</blockquote>
5. Запустите приложение командой: <blockquote>uvicorn main:app --reload</blockquote>
6. Запустите локальное приложение Postman, либо перейдите по <a href="https://web.postman.co/">ссылке</a> и создайте новый запрос.
7. Выполните один из запросов.

## Принимаемые запросы:
1. Проверить готовность приложения:<br><blockquote>GET>> http://127.0.0.1:8000/status</blockquote>
2. Узнать версию приложения и метаданные:<br><blockquote>GET>> http://127.0.0.1:8000/version</blockquote>
3. Получить предсказание модели:<br><blockquote>POST>> http://127.0.0.1:8000/predict<br></blockquote>

<i>PS: для последнего запроса передайте один из json'ов из папки 'Samples'. Для этого выберете 'Body -> raw -> JSON -> вставьте данные в поле.</i>

## Технологии
![pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Pydantic](https://img.shields.io/badge/Pydantic-%23ffffff.svg?style=for-the-badge&logo=Pydantic&logoColor=red)
![Pipeline](https://img.shields.io/badge/Pipeline-%23ffffff.svg?style=for-the-badge&logo=Pipeline&logoColor=red)
![Seaborn](https://img.shields.io/badge/Seaborn-%23ffffff.svg?style=for-the-badge&logo=Seaborn&logoColor=red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
