from sentosadl.model.modelview import ModelView
from sentosadl.data.dataview import DataView
import tensorflow as tf
print("----------------begin-----------------")
dataView = DataView("zhaozhenchong")
modelMarket = ModelMarket("zhaozhenchong")
# 读取数据视图
image_label_ds = dataView.load_data_view("flower_photos")
# 数据预处理
def pre_process(image_label):
    # 训练的基本方法
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 32
    image_count = 3670
    # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
    # 被充分打乱。
    ds = image_label.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
# Picture preprocessing function
def preprocess_image((image,label)):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return (image,label)
ds=image_label_ds.map(preprocess_image)
ds = pre_process(ds)
# 传递数据集至模型
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable = False
# 生成模型
model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation='softmax')])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

# 训练模型
model.fit(ds, epochs=1, steps_per_epoch=3)
# 模型保存
modelMarket.save_model_view(model, "tarinModel")
# 载入模型 它返回一个对象，其中包含可用于进行推断的函数
modeltest = modelMarket.load_model_view("tarinModel")
# 预测数据集
DEFAULT_FUNCTION_KEY = "serving_default"
inference_func = modeltest.signatures[DEFAULT_FUNCTION_KEY]
predict_dataset = ds.map(lambda image, label: image)

for batch in predict_dataset.take(1):
    result = inference_func(batch)
# 保存预测结果
dataset = tf.data.Dataset.from_tensor_slices(result)
dataView.save_data_view("zzc_output", dataset)
print("----------------end-----------------")
