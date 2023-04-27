var interactionStatisticsPublisher;
var playSignalPublisher;
var selectedChoicesPublisher;

var robot_IP;
var ros;

var state_dict = {
    'visual': false,
    'audio': false,
    'kinesthetic': false,
    'visual_choice': -1,
    'audio_choice': -1,
    'kinesthetic_choice': -1,
}

var interaction_metrics = {
    'moreOptions_count': 0, 
    'test_count': 0,
}

function getEmptyState() {
    return {'visual': false,'audio': false,'kinesthetic': false,
    'visual_choice': -1,'audio_choice': -1, 'kinesthetic_choice': -1}
}

function processVisualSelection(id) {
    //Update state of the interface
    let choice_num = parseInt(id.split('*')[1])
    state_dict.visual = choice_num < 0 ? false: true
    state_dict.visual_choice = choice_num

    //Send the specific stimulus
    state_dict_copy = getEmptyState()
    state_dict_copy.visual = choice_num < 0 ? false: true
    state_dict_copy.visual_choice = choice_num
    console.log(state_dict_copy)


} 

function processAudioSelection(id) {
    //Update state of the interface
    let choice_num = parseInt(id.split('*')[1])
    state_dict.audio = choice_num < 0 ? false: true
    state_dict.audio_choice = choice_num

    //Send the specific stimulus
    state_dict_copy = getEmptyState()
    state_dict_copy.audio = choice_num < 0 ? false: true
    state_dict_copy.audio_choice = choice_num
    console.log(state_dict_copy)


} 

function processKinesteticSelection(id) {
    //Update state of the interaction
    let choice_num = parseInt(id.split('*')[1])
    state_dict.kinesthetic = choice_num < 0 ? false: true
    state_dict.kinesthetic_choice = choice_num

    //Send the specific interaction
    state_dict_copy = getEmptyState()
    state_dict_copy.kinesthetic = choice_num < 0 ? false: true
    state_dict_copy.kinesthetic_choice = choice_num
    console.log(state_dict_copy)


} 

function processTestSelection(id) {
    //Update state of the interaction
    let count_num = interaction_metrics.test_count
    count_num = count_num + 1,
    interaction_metrics.test_count = count_num

    //Send interaction metrics
    console.log(interaction_metrics)
} 

function processMoreOptionsSelection(id) {
  //Update state of the interaction
  let count_num = interaction_metrics.moreOptions_count
  count_num = count_num + 1,
  interaction_metrics.moreOptions_count = count_num

  //Send interaction metrics
  console.log(interaction_metrics)
} 

function processFinalizeSelection(id) {
  //send the interaction data for logging
  send_interaction_metrics()

  //Update state of the interaction
  interaction_metrics.test_count = 0
  interaction_metrics.moreOptions_count = 0

  //Send interaction metrics
  console.log(interaction_metrics)
} 

function togglePopup() {
    document.getElementById("popup-1").classList.toggle("active");
}



// var pathProposalActionServer;

/*
Function for sending the goal to the action server
*/




//Sends play signal message to play signal publisher
function send_play_signal(message){
  var feedback = new ROSLIB.Message({
    visual : message.visual,
    visual_choice : message.visual_choice,

    audio : message.audio,
    audio_choice: message.audio_choice,

    kinesthetic : message.kinesthetic,
    kinesthetic_choice: message.kinesthetic_choice,
  });

  playSignalPublisher.publish(feedback)
}

function send_selected_choices(){
  var feedback = new ROSLIB.Message({
    visual : state_dict.visual,
    visual_choice : state_dict.visual_choice,

    audio : state_dict.audio,
    audio_choice: state_dict.audio_choice,

    kinesthetic : state_dict.kinesthetic,
    kinesthetic_choice: state_dict.kinesthetic_choice,
  });

  selectedChoicesPublisher.publish(feedback)
}

function send_interaction_metrics(){
  var feedback = new ROSLIB.Message({
    
    num_more_options_presses : interaction_metrics.moreOptions_count, 
    num_test_presses : interaction_metrics.test_count
    
  });
  console.log(feedback)
  interactionStatisticsPublisher.publish(feedback)
}


window.onload = function () {
    // determine robot address automatically
    // robot_IP = location.hostname;
    // set robot address statically
    robot_IP = "0.0.0.0";

    // // Init handle for rosbridge_websocket
    ros = new ROSLIB.Ros({
        url: "ws://" + robot_IP + ":9090"
    });

    
      
      //playSignal topic
      playSignalPublisher = new ROSLIB.Topic({
          ros : ros,
          name: '/play_signal',
          messageType : 'interface_msgs/State'
      });

      selectedChoicesPublisher = new ROSLIB.Topic({
        ros : ros,
        name: '/selected_choices',
        messageType : 'interface_msgs/State'
      });

      interactionStatisticsPublisher = new ROSLIB.Topic({
        ros : ros,
        name: '/interaction_statistics',
        messageType : 'interface_msgs/ButtonPresses'
      });

      

    // pathProposalActionServer = new ROSLIB.ActionClient({
    //     ros : ros,
    //     serverName : '/path_proposal',
    //     actionName : 'path_proposal/PathRequestAction'
    // });

    
      ros.on('connection', function() {
        console.log('Connected to websocket server.');
      });
    
      ros.on('error', function(error) {
        console.log('Error connecting to websocket server: ', error);
      });
    
      ros.on('close', function() {
        console.log('Connection to websocket server closed.');
      });

    // get handle for video placeholder
    video = document.getElementById('video');

    // Populate video source 
    video.src = "http://" + robot_IP + ":8080/stream?topic=/gui_video_stream&type=mjpeg&quality=80";
    video.onload = function () {
        // load other visual aspects
        
    };

    document.getElementById('video').addEventListener("click", generate_path_at_point, false);
}