namespace Neural_Network{
    public struct DataPoint{
        public readonly double[] Inputs;
        public readonly double[] ExpectedOutputs;
        public readonly int Label;

        public DataPoint(double[] inputs, int label, int numberOfLabels){
            Inputs = inputs;
            Label = label;
            ExpectedOutputs = CreateOneHot(label, numberOfLabels);
        }

        private static double[] CreateOneHot(int index, int number){
            double[] oneHot = new double[number];
            oneHot[index] = 1;
            return oneHot;
        }
    }
}