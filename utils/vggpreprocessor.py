def vgg_preprocessor(image_sample):
    img = image.load_img(image_sample, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    result = tf.keras.applications.resnet.preprocess_input(img_array_expanded_dims)

    return result
