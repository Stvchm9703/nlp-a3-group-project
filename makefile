run-transformer-prepare:
	if ![[ -d datasets ]]; then \
		mkdir datasets; \
	fi;\
	if ![[ -d datasets/glove_embeddings ]]; then\
		mkdir datasets/glove_embeddings; \
	fi;
	if ![[ -f  datasets/glove_embeddings/glove.6B.zip ]]; then\
		curl https://downloads.cs.stanford.edu/nlp/data/glove.6B.zips -o datasets/glove_embeddings/glove.6B.zip; \
		unzip -j datasets/glove_embeddings/glove.6B.zip glove.6B.300d.txt -d datasets/glove_embeddings; \
	fi; \
	python src/prepare.py