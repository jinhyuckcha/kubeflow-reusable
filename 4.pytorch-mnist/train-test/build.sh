registry=ckwlsgur20
img_name=mnist
ver=4.0v

docker build -t "${img_name}:$ver" .
docker tag "${img_name}:$ver" "${registry}/${img_name}:$ver"
docker push "${registry}/${img_name}:{$ver}"