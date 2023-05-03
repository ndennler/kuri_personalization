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
  var visualStimPub = new ROSLIB.Topic({
    ros : ros,
    name : '/set_visual_stimulus',
    messageType : 'std_msgs/Int8'
  });

  var auditoryStimPub = new ROSLIB.Topic({
    ros : ros,
    name : '/set_auditory_stimulus',
    messageType : 'std_msgs/Int8'
  });

  var kinestheticStimPub = new ROSLIB.Topic({
    ros : ros,
    name : '/set_kinesthetic_stimulus',
    messageType : 'std_msgs/Int8'
  });

  var playSignalPub = new ROSLIB.Topic({
    ros : ros,
    name : '/play_signal',
    messageType : 'std_msgs/String'
  });


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
            let new_stimulus_index = query[parseInt(element.id.slice(-1)) -1]
            element.innerHTML = ""
            element.style.backgroundImage = `url("${video_ims[new_stimulus_index]}")`;
            element.dataset.index = new_stimulus_index
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
            let new_stimulus_index = query[parseInt(element.id.slice(-1)) -1]
            element.style.backgroundImage = `url("${audio_ims[new_stimulus_index].file}")`;
            element.innerHTML = audio_ims[new_stimulus_index].name.replace(/_/g, ' ')
            element.dataset.index = new_stimulus_index
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
            let new_stimulus_index = query[parseInt(element.id.slice(-1)) -1]
            element.style.backgroundImage = `url("${movement_ims[new_stimulus_index]}")`;
            element.innerHTML = ''
            element.dataset.index = new_stimulus_index
        }
    });
  });








  var choice = new ROSLIB.Message({
    data : 1
  });

//   kinestheticChoicePub.publish(choice);
//   auditoryChoicePub.publish(choice);
//   visualChoicePub.publish(choice);