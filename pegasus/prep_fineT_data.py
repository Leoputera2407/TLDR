import pandas as pd
import tensorflow as tf
import json

save_path = "pegasus/data/testdata/fine_tune_data.tfrecords"
with open('all_v1.json') as f:
    data = json.loads(f)

inputs = [data[f"legalsum{i+1 if i+1 < 10 else '0'+ (i+1)}"]["original_text"] for i in range(422)]
targets = [data[f"legalsum{i+1 if i+1 < 10 else '0'+ (i+1)}"]["reference_summary"] for i in range(422)]

input_dict = dict(
                  inputs=inputs,
                  targets=targets
                 )

data = pd.DataFrame(input_dict)

with tf.io.TFRecordWriter(save_path) as writer:
    for row in data.values:
        inputs, targets = row[:-1], row[-1]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "inputs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[inputs[0].encode('utf-8')])),
                    "targets": tf.train.Feature(bytes_list=tf.train.BytesList(value=[targets.encode('utf-8')])),
                }
            )
        )
        writer.write(example.SerializeToString())

