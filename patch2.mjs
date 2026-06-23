import fs from 'fs';

const filePath = 'src/simulation/realtime/RealtimeSimulation.ts';
let code = fs.readFileSync(filePath, 'utf8');

code = code.replace(/this\.forceBindGroups\[this\.currentBufferIndex\]s = \[/g, 'this.forceBindGroups = [');

fs.writeFileSync(filePath, code);
