package io.github.introml.activityrecognition;

import android.content.Context;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class TensorFlowClassifier {
    //get the TensorFlow Java API
    static {
        System.loadLibrary("tensorflow_inference");
    }
    //
    private TensorFlowInferenceInterface inferenceInterface;
    // load the Exported TF model
//    private static final String MODEL_FILE = "file:///android_asset/frozen_model.pb"; //specify the address to the TF model
    private static final String MODEL_FILE = "file:///android_asset/frozen_har.pb"; // this is an alternative model file
    private static final String INPUT_NODE = "input_1"; //what do the input and output node stand for? Ans: it is the node name in the TF.pd model
    private static final String[] OUTPUT_NODES = {"output_1"}; //output node name in the TF.pd
    private static final String OUTPUT_NODE = "output_1";//output node name in the TF.pd
    private static final long[] INPUT_SIZE = {1, 200, 3};// input tensor shape :TBD
    private static final int OUTPUT_SIZE = 6; // output probabilities: TBD

    public TensorFlowClassifier(final Context context) {
        inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), MODEL_FILE);
    }

    public float[] predictProbabilities(float[] data) {
        float[] result = new float[OUTPUT_SIZE];
        inferenceInterface.feed(INPUT_NODE, data, INPUT_SIZE);
        inferenceInterface.run(OUTPUT_NODES);
        inferenceInterface.fetch(OUTPUT_NODE, result);
        return result; //return the probabilities of each potential class
    }
}
