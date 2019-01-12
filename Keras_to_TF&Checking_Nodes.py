from keras.models import Model
from keras.layers import *
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
os.chdir("D:\Academic\\Research Works\\Activity Evaluation and DSS\\Mobile app")
#Convert Trained Keras Model into TF
def keras_to_tensorflow(keras_model, output_dir, model_name,out_prefix="output_", log_tensorboard=True):

    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    out_nodes = []

    for i in range(len(keras_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(keras_model.output[i], out_prefix + str(i + 1))

    sess = K.get_session()

    from tensorflow.python.framework import graph_util, graph_io

    init_graph = sess.graph.as_graph_def()

    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)

    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard

        import_pb_to_tensorboard.import_to_tensorboard(
            os.path.join(output_dir, model_name),
            output_dir)
#Define A Keras Model
def ConvLSTM(input_shape=(20,30,1)):# HAR dataset dimension
    image_input = Input(shape=input_shape)
    #Four Convolutional Layers
    network = Conv2D(64, kernel_size=(5, 30), strides=(1, 1),padding='same')(image_input)
    network = Activation("tanh")(network)
    network = Conv2D(64, kernel_size=(5, 30), strides=(1, 1),padding='same')(network)
    network = Activation("tanh")(network)
    network = Conv2D(64, kernel_size=(5, 30), strides=(1, 1),padding='same')(network)
    network = Activation("tanh")(network)
    network = Conv2D(64, kernel_size=(5, 30), strides=(1, 1),padding='same')(network)
    network = Activation("tanh")(network)
    #turn the multilayer tensor into single layer tensor
    network = Reshape((20, -1),input_shape=(20,30,64))(network)
    network = Dropout(0.5)(network)
    network = LSTM(128, return_sequences=True, input_shape=(20, 1920))(network)
    network = Dropout(0.5)(network)
    network = LSTM(128)(network)
    network = Dense(7)(network) #Dense Layer of number of classes.
    network = Activation("softmax",name="output")(network)
    input_image = image_input
    model = Model(inputs=input_image, outputs=network)
    return model
#Load a trained keras model
keras_model = ConvLSTM()
keras_model.load_weights("2D_CNN_LSTM_checkpoint(F1)_0.h5")
output_dir = os.path.join(os.getcwd(),"checkpoint")
keras_to_tensorflow(keras_model,output_dir=output_dir,model_name="2D_CNN_LSTM.pb")
print("MODEL SAVED")
#Load the model for checking the frozen model
GRAPH_PB_PATH = 'D:\Academic\\Research Works\\Activity Evaluation and DSS\\Mobile app\\2D_CNN_LSTM.pb'
def get_IO_Nodes_Names(GRAPH_PB_PATH):
   with tf.Session() as sess:
       print("load graph")
       with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
           graph_def = tf.GraphDef()
       graph_def.ParseFromString(f.read())
       sess.graph.as_default()
       tf.import_graph_def(graph_def, name='')
       graph_nodes=[n for n in graph_def.node]
       print(graph_nodes[0])
       print(graph_nodes[-1])

get_IO_Nodes_Names(GRAPH_PB_PATH)