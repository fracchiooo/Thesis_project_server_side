import { Device } from "./Device";
import { Prediction } from "./Prediction";

export interface User {
    username: string;
    devices?: Device[];
    predictions?: Prediction[];
  }