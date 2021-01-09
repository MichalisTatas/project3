run-reduce:
	python reduce.py -d data/t10kimages -q queryset --od output_dataset --oq output_queryset

run-emd:
	python3 emd.py -d data/trainimages -q data/t10kimages --l1 data/trainlabels --l2 data/t10klabels -o output_file --EMD t

clean:
	rm -rf __pycache__
	rm output_dataset
	rm output_queryset