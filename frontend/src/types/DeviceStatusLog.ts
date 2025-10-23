import { Device } from "./Device";

export interface DeviceStatusLog {
    id: Number;
    device: Device;
    statusDate?: Date;
    temperature?: Number;
    frequency?: Number;
}
