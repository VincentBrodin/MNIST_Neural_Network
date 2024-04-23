using static System.Math;

namespace Neural_Network{
    public class Layer{
        public readonly int NumberOfNodesIn;
        public readonly int NumberOfNodesOut;

        public readonly double[] Weights;
        public readonly double[] Biases;

        // Cost gradient with respect to weights and with respect to biases
        public readonly double[] CostGradientW;
        public readonly double[] CostGradientB;

        // Used for adding momentum to gradient descent
        public readonly double[] WeightVelocities;
        public readonly double[] BiasVelocities;

        public IActivation Activation;

        public Layer(int numberOfNodesIn, int numberOfNodesOut, System.Random rng){
            NumberOfNodesIn = numberOfNodesIn;
            NumberOfNodesOut = numberOfNodesOut;
            Activation = new Activation.Sigmoid();

            Weights = new double[numberOfNodesIn * numberOfNodesOut];
            CostGradientW = new double[Weights.Length];
            Biases = new double[numberOfNodesOut];
            CostGradientB = new double[Biases.Length];

            WeightVelocities = new double[Weights.Length];
            BiasVelocities = new double[Biases.Length];

            InitializeRandomWeights(rng);
        }

        public double[] CalculateOutputs(double[] inputs){
            double[] weightedInputs = new double[NumberOfNodesOut];

            for (int nodeOut = 0; nodeOut < NumberOfNodesOut; nodeOut++){
                double weightedInput = Biases[nodeOut];

                for (int nodeIn = 0; nodeIn < NumberOfNodesIn; nodeIn++){
                    weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
                }

                weightedInputs[nodeOut] = weightedInput;
            }

            double[] activations = new double[NumberOfNodesOut];
            for (int outputNode = 0; outputNode < NumberOfNodesOut; outputNode++){
                activations[outputNode] = Activation.Activate(weightedInputs, outputNode);
            }

            return activations;
        }

        public double[] CalculateOutputs(double[] inputs, LayerLearnData learnData){
            learnData.Inputs = inputs;

            for (int nodeOut = 0; nodeOut < NumberOfNodesOut; nodeOut++){
                double weightedInput = Biases[nodeOut];
                for (int nodeIn = 0; nodeIn < NumberOfNodesIn; nodeIn++){
                    weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
                }

                learnData.WeightedInputs[nodeOut] = weightedInput;
            }

            for (int i = 0; i < learnData.Activations.Length; i++){
                learnData.Activations[i] = Activation.Activate(learnData.WeightedInputs, i);
            }

            return learnData.Activations;
        }

        public void ApplyGradients(double learnRate, double regularization, double momentum){
            double weightDecay = (1 - regularization * learnRate);

            for (int i = 0; i < Weights.Length; i++){
                double weight = Weights[i];
                double velocity = WeightVelocities[i] * momentum - CostGradientW[i] * learnRate;
                WeightVelocities[i] = velocity;
                Weights[i] = weight * weightDecay + velocity;
                CostGradientW[i] = 0;
            }


            for (int i = 0; i < Biases.Length; i++){
                double velocity = BiasVelocities[i] * momentum - CostGradientB[i] * learnRate;
                BiasVelocities[i] = velocity;
                Biases[i] += velocity;
                CostGradientB[i] = 0;
            }
        }

        public void CalculateOutputLayerNodeValues(LayerLearnData layerLearnData, double[] expectedOutputs, ICost cost){
            for (int i = 0; i < layerLearnData.NodeValues.Length; i++){
                // Evaluate partial derivatives for current node: cost/activation & activation/weightedInput
                double costDerivative = cost.CostDerivative(layerLearnData.Activations[i], expectedOutputs[i]);
                double activationDerivative = Activation.Derivative(layerLearnData.WeightedInputs, i);
                layerLearnData.NodeValues[i] = costDerivative * activationDerivative;
            }
        }

        public void CalculateHiddenLayerNodeValues(LayerLearnData layerLearnData, Layer oldLayer,
            double[] oldNodeValues){
            for (int newNodeIndex = 0; newNodeIndex < NumberOfNodesOut; newNodeIndex++){
                double newNodeValue = 0;
                for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.Length; oldNodeIndex++){
                    double weightedInputDerivative = oldLayer.GetWeight(newNodeIndex, oldNodeIndex);
                    newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
                }

                newNodeValue *= Activation.Derivative(layerLearnData.WeightedInputs, newNodeIndex);
                layerLearnData.NodeValues[newNodeIndex] = newNodeValue;
            }
        }

        public void UpdateGradients(LayerLearnData layerLearnData){
            lock (CostGradientW){
                for (int nodeOut = 0; nodeOut < NumberOfNodesOut; nodeOut++){
                    double nodeValue = layerLearnData.NodeValues[nodeOut];
                    for (int nodeIn = 0; nodeIn < NumberOfNodesIn; nodeIn++){
                        double derivativeCost = layerLearnData.Inputs[nodeIn] * nodeValue;
                        CostGradientW[GetFlatWeightIndex(nodeIn, nodeOut)] += derivativeCost;
                    }
                }
            }

            lock (CostGradientB){
                for (int nodeOut = 0; nodeOut < NumberOfNodesOut; nodeOut++){
                    double derivativeCost = 1 * layerLearnData.NodeValues[nodeOut];
                    CostGradientB[nodeOut] += derivativeCost;
                }
            }
        }

        public double GetWeight(int nodeIn, int nodeOut){
            int flatIndex = nodeOut * NumberOfNodesIn + nodeIn;
            return Weights[flatIndex];
        }

        public int GetFlatWeightIndex(int inputNeuronIndex, int outputNeuronIndex){
            return outputNeuronIndex * NumberOfNodesIn + inputNeuronIndex;
        }

        public void SetActivationFunction(IActivation activation){
            this.Activation = activation;
        }

        public void InitializeRandomWeights(System.Random random){
            for (int i = 0; i < Weights.Length; i++){
                Weights[i] = RandomInNormalDistribution(random, 0, 1) / Sqrt(NumberOfNodesIn);
            }
        }

        private double RandomInNormalDistribution(System.Random random, double mean, double standardDeviation){
            double x1 = 1 - random.NextDouble();
            double x2 = 1 - random.NextDouble();

            double y1 = Sqrt(-2.0 * Log(x1)) * Cos(2.0 * PI * x2);
            return y1 * standardDeviation + mean;
        }
    }
}