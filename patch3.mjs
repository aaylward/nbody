import fs from 'fs';

const filePath = 'src/simulation/realtime/RealtimeSimulation.ts';
let code = fs.readFileSync(filePath, 'utf8');

code = code.replace(/this\.kickDriftBindGroups\[this\.currentBufferIndex\]s = \[/g, 'this.kickDriftBindGroups = [');
code = code.replace(/this\.kickBindGroups\[this\.currentBufferIndex\]s = \[/g, 'this.kickBindGroups = [');
code = code.replace(/this\.interpolateBindGroups\[this\.currentBufferIndex\]s = \[/g, 'this.interpolateBindGroups = [');

fs.writeFileSync(filePath, code);
