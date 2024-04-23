using System;
using System.Collections.Generic;
using System.Linq;

namespace Neural_Network{
    public class NeuralNetwork{
        public readonly Layer[] Layers;
        public readonly int[] LayerSizes;

        public ICost Cost;
        private NetworkLearnData[] _batchLearnData;

        public NeuralNetwork(params int[] layerSizes){
            LayerSizes = layerSizes;
            Random random = new();

            Layers = new Layer[layerSizes.Length - 1];
            for (int i = 0; i < Layers.Length; i++){
                Layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], random);
            }

            Cost = new Cost.MeanSquaredError();
        }

        public (int predictedClass, double[] outputs) Classify(double[] inputs){
            double[] outputs = CalculateOutputs(inputs);
            int predictedClass = MaxValueIndex(outputs);
            return (predictedClass, outputs);
        }

        public double[] CalculateOutputs(double[] inputs){
            return Layers.Aggregate(inputs, (current, layer) => layer.CalculateOutputs(current));
        }


        public void Learn(DataPoint[] trainingData, double learnRate, double regularization = 0, double momentum = 0){
            if (_batchLearnData == null || _batchLearnData.Length != trainingData.Length){
                _batchLearnData = new NetworkLearnData[trainingData.Length];
                for (int i = 0; i < _batchLearnData.Length; i++){
                    _batchLearnData[i] = new NetworkLearnData(Layers);
                }
            }

            System.Threading.Tasks.Parallel.For(0, trainingData.Length,
                (i) => { UpdateGradients(trainingData[i], _batchLearnData[i]); });


            foreach (Layer layer in Layers){
                layer.ApplyGradients(learnRate / trainingData.Length, regularization, momentum);
            }
        }


        private void UpdateGradients(DataPoint data, NetworkLearnData learnData){
            double[] inputsToNextLayer = data.Inputs;

            for (int i = 0; i < Layers.Length; i++){
                inputsToNextLayer = Layers[i].CalculateOutputs(inputsToNextLayer, learnData.LayerData[i]);
            }

            int outputLayerIndex = Layers.Length - 1;
            Layer outputLayer = Layers[outputLayerIndex];
            LayerLearnData outputLearnData = learnData.LayerData[outputLayerIndex];

            outputLayer.CalculateOutputLayerNodeValues(outputLearnData, data.ExpectedOutputs, Cost);
            outputLayer.UpdateGradients(outputLearnData);

            for (int i = outputLayerIndex - 1; i >= 0; i--){
                LayerLearnData layerLearnData = learnData.LayerData[i];
                Layer hiddenLayer = Layers[i];

                hiddenLayer.CalculateHiddenLayerNodeValues(layerLearnData, Layers[i + 1],
                    learnData.LayerData[i + 1].NodeValues);
                hiddenLayer.UpdateGradients(layerLearnData);
            }
        }

        public void SetCostFunction(ICost costFunction){
            Cost = costFunction;
        }

        public void SetActivationFunction(IActivation activation){
            SetActivationFunction(activation, activation);
        }

        public void SetActivationFunction(IActivation activation, IActivation outputLayerActivation){
            for (int i = 0; i < Layers.Length - 1; i++){
                Layers[i].SetActivationFunction(activation);
            }

            Layers[^1].SetActivationFunction(outputLayerActivation);
        }


        public int MaxValueIndex(double[] values){
            double maxValue = double.MinValue;
            int index = 0;
            for (int i = 0; i < values.Length; i++){
                if (values[i] > maxValue){
                    maxValue = values[i];
                    index = i;
                }
            }

            return index;
        }
    }


    public class NetworkLearnData{
        public readonly LayerLearnData[] LayerData;

        public NetworkLearnData(IReadOnlyList<Layer> layers){
            LayerData = new LayerLearnData[layers.Count];
            for (int i = 0; i < layers.Count; i++){
                LayerData[i] = new LayerLearnData(layers[i]);
            }
        }
    }

    public class LayerLearnData{
        public double[] Inputs;
        public readonly double[] WeightedInputs;
        public readonly double[] Activations;
        public readonly double[] NodeValues;

        public LayerLearnData(Layer layer){
            WeightedInputs = new double[layer.NumberOfNodesOut];
            Activations = new double[layer.NumberOfNodesOut];
            NodeValues = new double[layer.NumberOfNodesOut];
        }
    }
}