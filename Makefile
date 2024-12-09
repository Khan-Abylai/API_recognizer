build_api:
	docker build --build-arg LOCALE=C.UTF-8 -t doc.ark.ru/doc_recognizer_api:app api/

run:
	docker-compose up -d --build

stop:
	docker-compose down

restart:
	make stop
	make run

