frozen_model_dir = "models/resnet50_v1.5_fp32.pb"


graph = load_graph(frozen_model_dir)
input_tensor = graph.get_tensor_by_name("input_tensor:0")
output_tensor = graph.get_tensor_by_name("softmax_tensor:0")
sess = tf.compat.v1.Session(graph=graph)

image_net = ImageNet()
preprocessed_image = image_net.get_input_tensor(32, (224,224), pre_process_vgg)

start = time.time()
result = sess.run(output_tensor, feed_dict={input_tensor: preprocessed_image})
end = time.time()
total_time += (end - start)