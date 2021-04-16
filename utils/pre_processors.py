def pre_process_ssd(input_array):
    input_array = input_array.astype("float32")
    input_array *= (2.0 / 255.0)
    input_array -= 1.0
    # kinda equivalent solution:
    # input_array -= 127.5
    # input_array *= 0.007843
    return input_array
