# mle-template-case-sprint2

Добро пожаловать в репозиторий-шаблон Практикума для проекта 2 спринта. Ваша цель — улучшить ключевые метрики модели для предсказания стоимости квартир Яндекс Недвижимости.

Полное описание проекта хранится в уроке «Проект. Улучшение baseline-модели» на учебной платформе.

Здесь укажите имя вашего бакета: ```s3-student-mle-20240228-2fd44f5a96```

# Установка репозитория и зависимостей

```
git clone git@github.com:FrostyFrost/mle-project-sprint-2-v001.git
cd mle-project-sprint-2-v001

sudo apt-get update
sudo apt-get install python3.10-venv
python3.10 -m venv .venv_sprint02
source .venv_sprint02/bin/activate

pip install -r requirements.txt 
```
# Установка mlflow

```
export $(cat .env | xargs)
sh mlflow_server/server_start.sh 
```
Для регистрации первой версии модели запустить скрипт 
```
python mlflow_server/register_model.py 
```
В проекте модель зарегистрирована под именем:
```REGISTRY_MODEL_NAME = "estate_model"```



