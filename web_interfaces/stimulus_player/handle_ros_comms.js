var ros = new ROSLIB.Ros({
    url : 'ws://localhost:9090'
  });

  ros.on('connection', function() {
    console.log('Connected to websocket server.');
  });

  ros.on('error', function(error) {
    console.log('Error connecting to websocket server: ', error);
  });

  ros.on('close', function() {
    console.log('Connection to websocket server closed.');
  });

  //setup publishers
  //None teehee

// setup subscribers

var visualSetterReceiver = new ROSLIB.Topic({
    ros : ros,
    name : '/set_visual_stimulus',
    messageType : 'std_msgs/Int32'
  });

  visualSetterReceiver.subscribe(function(message) {
    switch_video_from_index(parseInt(message.data))
  });

var auditorySetterReceiver = new ROSLIB.Topic({
    ros : ros,
    name : '/set_auditory_stimulus',
    messageType : 'std_msgs/Int32'
  });

  auditorySetterReceiver.subscribe(function(message) {
    switch_audio_from_index(parseInt(message.data))
  });

var signalPlayReceiver = new ROSLIB.Topic({
    ros : ros,
    name : '/play_signal',
    messageType : 'std_msgs/String'
  });

  signalPlayReceiver.subscribe(function(message) {
    if(message.data == 'auditory' || message.data == 'Audio'){
      play_audio()
    } else if (message.data == 'visual' || message.data == 'Video'){
      play_video()
    } else if(message.data == 'kinesthetic' || message.data == 'Movement'){
      //do nothing
    } else {
      synch_stimulus()
    }
    console.log(message)
  });








  var choice = new ROSLIB.Message({
    data : 1
  });

//   kinestheticChoicePub.publish(choice);
//   auditoryChoicePub.publish(choice);
//   visualChoicePub.publish(choice);