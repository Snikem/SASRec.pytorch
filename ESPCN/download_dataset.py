import kagglehub

# Download latest version
path = kagglehub.dataset_download("linijiafei/voc2012")

print("Path to dataset files:", path)