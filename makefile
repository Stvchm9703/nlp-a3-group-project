PYTHON=/usr/local/bin/python3

run-transformer-prepare:
	if ! [[ -d datasets ]]; then \
		mkdir datasets; \
	fi;\
	if ! [[ -d datasets/glove_embeddings ]]; then\
		mkdir datasets/glove_embeddings; \
	fi;
	if ! [[ -f  datasets/glove_embeddings/glove.6B.zip ]]; then\
		curl https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip -o datasets/glove_embeddings/glove.6B.zip; \
		unzip -j datasets/glove_embeddings/glove.6B.zip glove.6B.300d.txt -d datasets/glove_embeddings; \
	fi; \
	${PYTHON} src/prepare.py


run-transformer-test:
	${PYTHON} src/test_and_benchmark.py\
		--model_path=checkpoints/May-06_10-04-56/model_119.pth \
		--config=checkpoints/May-06_10-04-56/config.json ;\