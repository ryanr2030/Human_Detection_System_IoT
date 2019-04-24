
'use strict';
//iot hub con string
var connectionString = 'Insert Your Connection String';
//express app dec and instantiation
var app = require('express')();
//attach the app to the node server to be served up
var http = require('http').Server(app);
//attach socket io to the node server for client server sockets
var io = require('socket.io')(http);

//on connection log it
io.on('connection', function(socket){
  console.log('a user connected');
  //on count event to emit from client to server and server to client
  //when a message is received
  socket.on('count', function(msg){
  });
  socket.on('disconnect', function(){
    console.log('user disconnected');
  });
});

//Listen for http calls on port 3000
http.listen(3000, function(){
  console.log('listening on *:3000');
});


// The sample connects to an IoT hub's Event Hubs-compatible endpoint
// to read messages sent from a device.
var { EventHubClient, EventPosition } = require('@azure/event-hubs');

var printError = function (err) {
  console.log(err.message);
};

//emit the count event to all clients
var printMessage = function (message) {
  io.emit('count', message.body);
  console.log(JSON.stringify(message.body));
};
// Connect to the partitions on the IoT Hub's Event Hubs-compatible endpoint.
// This example only reads messages sent after this application started.
var ehClient;
EventHubClient.createFromIotHubConnectionString(connectionString).then(function (client) {
  console.log("Successully created the EventHub Client from iothub connection string.");
  ehClient = client;
  return ehClient.getPartitionIds();
}).then(function (ids) {
  console.log("The partition ids are: ", ids);
  return ids.map(function (id) {
    return ehClient.receive(id, printMessage, printError, { eventPosition: EventPosition.fromEnqueuedTime(Date.now()) });
  });
}).catch(printError);

app.get('/', function(req, res){
  res.sendFile(__dirname + '/MapChartJs.html')
});