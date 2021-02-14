.PHONY: data

data:
	python ./data/input/midi2img/midi2img.py ./data/input/midi/ ./data/input/img/A/

train:
	python src/main.py --name "main" --class_conditional --epochs 2 --batch_size 2 --patience 10 --learning_rate 0.001 --dropout_rate 0.25 --optimizer adamax --checkpoints ./src/checkpoints/ --training --output_number 1 ./data/input/img/ ./data/output/img/

test:
	python src/main.py --name "test" --checkpoints ./src/checkpoints/ --log_level INFO --output_number 2 ./data/input/img/ ./data/output/img/
