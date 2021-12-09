import tensorflow as tf
import os
from datetime import datetime

def print_prof():
  if os.getenv("AIO_PROFILER", 0):
    try:
      tf.DLS.print_profile_data()  
    except AttributeError:
      print("Non dls tf")

class TBTracer:
  def __init__(self):
    self.should_trace = os.getenv("TRACE", 0)
    if self.should_trace: 
      # Set up logging.
      options = tf.profiler.experimental.ProfilerOptions()
      stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
      self.logdir = 'logs/func/%s' % stamp
      tf.profiler.experimental.start(self.logdir, options = options)

  def write(self):
    if self.should_trace:
      tf.profiler.experimental.stop()
    print_prof()
