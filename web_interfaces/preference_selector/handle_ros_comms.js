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

  var visualChoicePub = new ROSLIB.Topic({
    ros : ros,
    name : '/visual_choice',
    messageType : 'std_msgs/Int8'
  });

  var auditoryChoicePub = new ROSLIB.Topic({
    ros : ros,
    name : '/auditory_choice',
    messageType : 'std_msgs/Int8'
  });

  var kinestheticChoicePub = new ROSLIB.Topic({
    ros : ros,
    name : '/kinesthetic_choice',
    messageType : 'std_msgs/Int8'
  });

  var signalDonePub = new ROSLIB.Topic({
    ros : ros,
    name : '/signal_done',
    messageType : 'std_msgs/String'
  });

// setup subscribers

var visualQueryReceiver = new ROSLIB.Topic({
    ros : ros,
    name : '/visual_query',
    messageType : 'std_msgs/String'
  });

  visualQueryReceiver.subscribe(function(message) {
    var query = message.data.split(',')

    document.querySelectorAll(`[id*="Video"]`).forEach(element => {
        
        if(parseInt(element.id.slice(-1))){
            let new_element = query[parseInt(element.id.slice(-1)) -1]
            element.innerHTML = "Video " + new_element
        }
    });
  });

var auditoryQueryReceiver = new ROSLIB.Topic({
    ros : ros,
    name : '/auditory_query',
    messageType : 'std_msgs/String'
  });

  auditoryQueryReceiver.subscribe(function(message) {
    var query = message.data.split(',')

    document.querySelectorAll(`[id*="Audio"]`).forEach(element => {
        
        if(parseInt(element.id.slice(-1))){
            let new_element = query[parseInt(element.id.slice(-1)) -1]
            element.innerHTML = "Audio " + new_element
        }
    });
  });

var kinestheticQueryReceiver = new ROSLIB.Topic({
    ros : ros,
    name : '/kinesthetic_query',
    messageType : 'std_msgs/String'
  });

  kinestheticQueryReceiver.subscribe(function(message) {

    var query = message.data.split(',')
    
    document.querySelectorAll(`[id*="Movement"]`).forEach(element => {
        
        if(parseInt(element.id.slice(-1))){
            let new_element = query[parseInt(element.id.slice(-1)) -1]
            element.innerHTML = "Movement " + new_element
        }
    });
  });








  var choice = new ROSLIB.Message({
    data : 1
  });

//   kinestheticChoicePub.publish(choice);
//   auditoryChoicePub.publish(choice);
//   visualChoicePub.publish(choice);