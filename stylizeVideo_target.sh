set -e
# get a carriage return into `cr`
cr=`echo $'\n.'`
cr=${cr%.}

# find out whether ffmpeg or avconv is installed on the system
FFMPEG=ffmpeg
command -v $FFMPEG >/dev/null 2>&1 || {
  FFMPEG=avconv
  command -v $FFMPEG >/dev/null 2>&1 || {
    echo >&2 "This script requires either ffmpeg or avconv installed.  Aborting."; exit 1;
  }
}

if [ "$#" -le 1 ]; then
   echo "Usage: ./stylizeVideo_target <path_to_video> <path_to_style_image>"
   exit 1
fi

# parse arguments
filename=$(basename "$1")
extension="${filename##*.}"
filename="${filename%.*}"
filename=v_target/${filename//[%]/x}
style_image=$2

style_filename=$(basename "$2")
style_extension="${style_filename##*.}"
style_filename="${style_filename%.*}"
style_filename=${style_filename//[%]/x}

# create output folder
mkdir -p $filename

backend=${backend:-nn}

# extract video to images
$FFMPEG -i $1 -vf scale='320:240' -framerate 5 ${filename}/frame_%04d.ppm

content_weight=10
style_weight=75
tv_weight=1e-3
temporal_weight=1e3
gpu=${gpu:-0}

# calculate optical flow
bash makeOptFlow.sh ./${filename}/frame_%04d.ppm ./${filename}/flow_320:240

# perform style transfer
th ./artistic_video.lua \
-content_pattern ${filename}/frame_%04d.ppm \
-flow_pattern ${filename}/flow_320:240/backward_[%d]_{%d}.flo \
-flowWeight_pattern ${filename}/flow_320:240/reliable_[%d]_{%d}.pgm \
-content_weight $content_weight \
-style_weight $style_weight \
-tv_weight $tv_weight \
-temporal_weight $temporal_weight \
-output_folder ${filename}/ \
-style_image $style_image \
-backend $backend \
-gpu $gpu \
-cudnn_autotune \
-number_format ${style_filename}-%04d \
-num_images 64

# create video from output images
$FFMPEG -i ${filename}/out-${style_filename}-%04d.png -framerate 5 ${filename}-stylized_${style_filename}.$extension

# save output images as npy
python save_stylized_npy.py \
--styled-video-name ${filename}-stylized_${style_filename}.$extension \
--orig-video-name ${filename} \
--style-img-name ${style_filename}
