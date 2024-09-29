# Demo
Для запуска демо нужно выполнить инициализацию milvus c помощью docker compose 
``` sh
docker compose up
```
Установить зависимости 
``` sh
python -m venv my_venv
source my_venv/bin/activate
python -m pip install -r requirements.txt
```
Заполнить milvus данными. В dataset.csv лежат эмбединги
python load_db.py
```
Запустить скрипт 
``` sh
python main.py
```
Для запроса можно использовать либо postman либо curl 
``` sh
curl -X POST http://127.0.0.1:8001/check-video-duplicate  -H "Content-Type: application/json" -d "{\"videoLink\":\"https://cdn-st.rutubelist.ru/media/41/92/c4d38e264067a28e579979d9fa6b/fhd.mp4\"}"
```
