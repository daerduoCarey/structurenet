objpath=$1
pngpath=$2
blender model.blend  --background --python renderBatch.py -- $objpath $pngpath > /dev/null
