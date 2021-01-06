run-bottleneck :
	python reduce.py -d data/t10kimages -q queryset --od output_dataset --oq output_queryset

clean:
	rm -rf __pycache__
	rm output_dataset
	rm output_queryset