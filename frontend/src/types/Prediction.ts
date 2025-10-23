export interface Prediction {
    id: number;
    timestamp: string;
    predictedConcentration: Number;
    predictedUncertainty: Number;
    observedConcentration?: Number | null;
    initialConcentration: Number;
    frequency: Number;
    dutyCycle: Number;
    timeLasted: Number;
    temperature: Number;
    sentToDataset?: boolean;
}