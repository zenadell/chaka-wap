import { io } from 'socket.io-client';
const socket = io('http://localhost:3000');

socket.on('connect', () => {
    console.log('Connected to test socket');
    
    socket.emit('list_sessions');
    socket.emit('get_status', 'temple');
});

socket.onAny((event, ...args) => {
    console.log(`[Event: ${event}]`, JSON.stringify(args));
    if (event === 'status' && args[0].sessionId === 'temple') {
        const data = args[0];
        if (data.status === 'connected') {
            if (!global.sentRefresh) {
                console.log('Sending refresh_connection...');
                socket.emit('refresh_connection', 'temple');
                global.sentRefresh = true;
            } else {
                console.log('SUCCESS: Session reconnected!');
                process.exit(0);
            }
        }
    }
});

setTimeout(() => {
    console.log('TIMEOUT: Did not reconnect in 15 seconds.');
    process.exit(1);
}, 25000);
