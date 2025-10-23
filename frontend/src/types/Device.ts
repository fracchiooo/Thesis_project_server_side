import { DeviceStatusLog } from "./DeviceStatusLog";
import { User } from "./User";

export interface Device {
    deviceEUI: string;
    lastUpdate?: Date;
    currentTemperature?: Number;
    currentSensedFrequency?: Number;
    logs?: DeviceStatusLog[];
    deviceEnvRequests?: Map<String, Object>;
    user: User;
}