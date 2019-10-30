import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import ops as utils_ops

import sys

sys.path.append('[modelsのパス]/research/object_detection')
from utils import label_map_util
from utils import visualization_utils as vis_util

# 学習済モデルの読み込み
PATH_TO_CKPT = 'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ラベルの読み込み
PATH_TO_LABELS = '[modelsへのパス]/research/object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# 画像の読み込みとnumpy配列への変換
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


filename = '[画像ファイルのパス]'
image = Image.open(filename)
image_np = load_image_into_numpy_array(image)

# セマンティックセグメンテーションの処理
with detection_graph.as_default():
    with tf.Session() as sess:
        # 入出力用テンソルのハンドルを取得
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        tensor_dict = {}
        tensor_dict['num_detections'] = tf.get_default_graph().get_tensor_by_name('num_detections:0')
        tensor_dict['detection_boxes'] = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
        tensor_dict['detection_scores'] = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
        tensor_dict['detection_classes'] = tf.get_default_graph().get_tensor_by_name('detection_classes:0')
        tensor_dict['detection_masks'] = tf.get_default_graph().get_tensor_by_name('detection_masks:0')

        # バッチ内の最初の画像の結果を取り出す
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

        # 各検出ボックスのマスクを画像全体上のマスクへ変換
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)

        # バッチ分の次元を追加
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)

        # 実行
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: np.expand_dims(image_np, 0)})

        # バッチ分の次元の削除と型変換
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

# 画像にマスクとバウンディングボックスを書き込んで出力
vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks'),
    use_normalized_coordinates=True,
    line_thickness=8)
Image.fromarray(image_np).save('out.png')
