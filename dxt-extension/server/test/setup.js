import { jest } from '@jest/globals';

// Mock child_process
jest.mock('child_process', () => ({
  spawn: jest.fn(),
  execSync: jest.fn()
}));

// Mock fs/promises
jest.mock('fs/promises', () => ({
  access: jest.fn(),
  readFile: jest.fn(),
  writeFile: jest.fn()
}));

// Mock ws (WebSocket)
jest.mock('ws', () => ({
  WebSocket: jest.fn()
}));

// Mock node-fetch
jest.mock('node-fetch', () => ({
  default: jest.fn()
}));

// Mock os
jest.mock('os', () => ({
  homedir: jest.fn(() => '/mock/home'),
  platform: jest.fn(() => 'darwin')
}));

// Mock path
jest.mock('path', () => ({
  join: jest.fn((...paths) => paths.join('/')),
  resolve: jest.fn((...paths) => paths.join('/'))
}));

// Mock process events
const originalProcess = global.process;
global.process = {
  ...originalProcess,
  env: {
    ...originalProcess.env,
    NODE_ENV: 'test'
  }
};

// Clean up after each test
afterEach(() => {
  jest.clearAllMocks();
});

// Global test timeout
jest.setTimeout(30000);